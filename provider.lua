dofile("/home/msmith/torchFunctions/csv.lua")
dofile("/home/msmith/torchFunctions/joinTables.lua")
dofile("/home/msmith/torchFunctions/shuffle.lua")

require "nn"
require "cunn"
require "image"

local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))

--[[
local trainCV = shuffle(csv.read("trainCV.csv"))
local testCV = shuffle(csv.read("trainCV.csv"))
local train = joinTables(trainCV,testCV) 
local test = csv.read("test.csv")
]]--

trainCVTemp = shuffle(csv.read("trainCV.csv"))
testCVTemp = shuffle(csv.read("trainCV.csv"))
trainCV = {}
testCV = {}
for i = 1, 100 do 
	trainCV[i] = trainCVTemp[i]
end
for i = 1, 20 do 
	testCV[i] = testCVTemp[i]
end

-- For main script
dataSizes = {}
dataSizes.trainCV = #trainCV
dataSizes.testCV = #testCV

--dataSizes.train = #train
--dataSizes.test = #test


Provider = torch.class 'Provider'
function Provider:__init(tid,nThreads,crossValidation)
	self.trainData = {
		data = {},
		labels = {},
		currentIdx = 1,
		nObs = 0,
		epoch = 1,
		finished = 0
	}
	self.testData = {
		data = {}, 
		labels = {},
		currentIdx = 1,
		nObs = 0,
		epoch = 1,
		finished = 0
	}
	local trainData = self.trainData
	local testData = self.testData
	function getxy(path)
		local obs = path:split(",")
		local dataPath = string.format("train/%s/%s",obs[2],obs[3]:gsub(".jpg","_x.jpg"))
		local label = tonumber(string.sub(obs[2],2,2))
		if label == 0 then label = 10 end
		return dataPath, label
	end
	if crossValidation == 1 then
		t1 = trainCV
		t2 = testCV
		for i = tid, #t2, nThreads do 
			x,y = getxy(t2[i])
			table.insert(testData.data,x)
			table.insert(testData.labels,y)
		end
	else 
		t1 = train 
		t2 = test 
		for i = tid, #t2, nThreads do 
			x = t2[i]
			table.insert(testData.data,x)
		end
		
	end
	local x,y
	for i = tid, #t1, nThreads do 
		x,y = getxy(t1[i])
		table.insert(trainData.data,x)
		table.insert(trainData.labels,y)
	end
	trainData.nObs = #trainData.data
	testData.nObs = #testData.data

	self.finishedTrainEpoch = 0
	self.finishedTestEpoch = 0

end

function augment(img)
	local aspectRatio = 640/480
	local cropX = torch.random(40)
	local cropY = torch.random(40/aspectRatio)
	local x2, y2 = img:size(3) - cropX, img:size(2) - cropY
	local dst = image.crop(x,cropX,cropY,x2,y2)
	return dst
end

function preprocess(img)
	--local yuv = image.rgb2yuv(img)
     	--yuv[1] = normalization(yuv[{{1}}])
	local dst = image.scale(img:squeeze(),params.inW,params.inH,"bilinear"):double()
	--dst:csub(dst:mean())
	return dst:resize(1,3,params.inW,params.inH)
end


function Provider:getBatch(trainOrTest)
	local X = {}
	local Y = {}
	local imgPaths = {}
	if trainOrTest == "train" then d = self.trainData else d = self.testData end

	local bs = params.batchSize

	if d.finished == 1 then d.finished = 2 return 0,0,0 
	elseif d.finished == 2 then return 0,0,0 end

	for i = d.currentIdx, math.min(d.currentIdx + bs - 1,d.nObs)  do

		path = d.data[i]
		x = image.loadJPG(path)
		x = preprocess(x)
		y = d.labels[i]
		table.insert(X,x)
		table.insert(Y,y)
		table.insert(imgPaths,path)
		d.currentIdx = d.currentIdx + 1 
		if d.currentIdx == d.nObs then 
			d.finished = 1
		end
	end
	X = torch.cat(X,1):cuda()
	Y = torch.Tensor(Y):cuda()
	collectgarbage()
	return imgPaths,X, Y
end

	
function example()
	 dofile("display.lua")
	 dofile("/home/msmith/torchFunctions/counter.lua")
	 params = {}
	 params.inW = 128
	 params.inH = 128
	 params.batchSize = 4 
	 prov1 = Provider.new(1,4,1)
	 prov2 = Provider.new(2,30,1)
	 counter = Counter.new()
	 for i =1, 200 do
		 imgPaths,X,Y = prov1:getBatch("test")
	 	--print(prov1.trainData.currentIdx,prov1.trainData.finished,prov1.trainData.nObs)
	 	print(prov1.testData.currentIdx,prov1.testData.finished,prov1.testData.nObs)
		 --display(X,0,0,"train")
	 end
 end
