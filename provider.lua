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
testCVTemp = shuffle(csv.read("testCV.csv"))
trainCV = {}
testCV = {}
for i = 1, 3000 do 
	trainCV[i] = trainCVTemp[i]
end
for i = 1, 300 do 
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
		local dataPath = string.format("train/%s/%s",obs[2],obs[3]:gsub(".jpg",".jpg"))
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

function Provider:estimateUV()
	X = {}
	local trdata = self.trainData
	for i = 1, trdata.nObs do 
		local x = image.loadJPG(trdata.data[i])
		local dst = image.scale(x,32,32,"bilinear"):double()
		local yuv = image.rgb2yuv(dst)	
		yuv:resize(1,yuv:size(1),yuv:size(2),yuv:size(3))
		X[i] = yuv
		xlua.progress(i,trdata.nObs)
		collectgarbage()
	end
	X = torch.cat(X,1)
	meanU = X:select(2,2):mean()
	stdU = X:select(2,2):std()
	meanV = X:select(2,3):mean()
	stdV = X:select(2,3):std()
	print(string.format("Mean/Std U = {%f,%f}.",meanU,stdU))
	print(string.format("Mean/Std V = {%f,%f}.",meanV,stdV))

end

function augment(img)
	local aspectRatio = 640/480
	local cropX = torch.random(40)
	local cropY = torch.random(40/aspectRatio)
	local x2, y2 = img:size(3) - cropX, img:size(2) - cropY
	local dst = image.crop(x,cropX,cropY,x2,y2)
	return dst
end

function channelSub(img)
	rMu,rStd = img:narrow(1,1,1):mean(), img:narrow(1,1,1):std()
	gMu,gStd = img:narrow(1,2,1):mean(), img:narrow(1,2,1):std()
	bMu,bStd = img:narrow(1,3,1):mean(), img:narrow(1,3,1):std()
	img[1] = img[1]:div(rStd)
	img[2] = img[2]:div(gStd)
	img[3] = img[3]:div(bStd)
end

function preprocess(img)
	local yuv = image.rgb2yuv(img)
	yuv = image.scale(yuv:squeeze(),params.inW,params.inH,"bilinear"):double()
     	yuv[1] = normalization(yuv[{{1}}])
	yuv:select(1,2):add(-0.006877)
	yuv:select(1,2):div(0.027300)
	yuv:select(1,3):add(0.042122)
	yuv:select(1,3):div(0.055548)
	return yuv:resize(1,3,params.inW,params.inH)
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
		if trainOrTest == "train" then
			x = augment(x)
		end
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
	 prov1 = Provider.new(1,1,1)
	 counter = Counter.new()
	 trainMeans = {}
	 testMeans = {}
	 --[[
	 for i =1, 4 do

		 imgPaths,X,Y = prov1:getBatch("train")
		 imgPaths,X,Y = prov1:getBatch("test")
		 print(imgPaths,Y)
	 	--print(prov1.trainData.currentIdx,prov1.trainData.finished,prov1.trainData.nObs)
		 --display(X,0,0,"train")
	 end
	 ]]--
 end
