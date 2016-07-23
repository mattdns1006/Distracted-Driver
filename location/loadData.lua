dofile("/home/msmith/torchFunctions/csv.lua")
dofile("/home/msmith/torchFunctions/joinTables.lua")
dofile("/home/msmith/torchFunctions/shuffle.lua")
require "xlua"

local cv = require "cv"
require "cv.imgcodecs"
require "cv.imgproc"

test = csv.read("../test.csv") -- Actual testing data
local train = csv.read("masks.csv") -- Training data for masks ~ 300 
local trainCV = csv.read("../trainCV.csv")
local testCV = csv.read("../testCV.csv")
local allTrain = joinTables(trainCV,testCV) -- Actual training data for prediction
loadData = {}

function loadData.init(tid,nThreads)
	local trainPaths = {}
	local allTrainPaths = {}
	local testPaths = {}
	for i = tid, #train, nThreads do
		table.insert(trainPaths,train[i])
	end
	for i = tid, #test, nThreads do
		table.insert(testPaths,test[i])
	end
	for i = tid, #allTrain, nThreads do
		table.insert(allTrainPaths,allTrain[i])
	end
	trainPaths = shuffle(trainPaths)
	testPaths = shuffle(testPaths)
	return trainPaths, testPaths, allTrainPaths
end

function loadData.loadImage(path)
	collectgarbage()
	local x = image.load(path,3):double()
	return x,y 
end

function loadData.crop(x,y,cropSize)
	local angle = torch.uniform(-0.1,0.1)
	local x = image.rotate(x,angle,"bilinear")
	local y = image.rotate(y,angle,"bilinear")
	local x1, y1 = torch.random(cropSize), torch.random(cropSize)
	local x2, y2 = x:size(3) - torch.random(cropSize), x:size(2) - torch.random(cropSize)
	local dstX = image.crop(x,x1,y1,x2,y2)
	local dstY = image.crop(y,x1,y1,x2,y2)
	return dstX,dstY
end

function loadData.rescale(x,w,h)
	local dstX = image.scale(x:squeeze(),w,h,'bilinear'):double()
	return dstX
end

function loadData.getData(trainOrTest,t,allTrain)
	local x,y
	local idx
	if trainIdx == nil or testIdx == nil then trainIdx , testIdx = 1,1 end
	if trainOrTest == "train" then idx = trainIdx; else idx = testIdx  end
	if trainOrTest == "train" then
		local yPath = t[idx]
		local yPathSp = yPath:split("/")
		local xPath = string.format("../train/%s/%s",yPathSp[3],yPathSp[4]:gsub(".bmp",".jpg"))
		x, y = image.loadJPG(xPath,3), cv.imread{yPath,0}
		x, y = loadData.crop(x,y,60)
		x = loadData.rescale(x,params.inW,params.inH)
		y = loadData.rescale(y,params.outW,params.outH)
		y:div(255)
		if trainIdx == #t then trainIdx = 1;  else trainIdx = trainIdx + 1; end
	elseif trainOrTest == "test" then
		local xPath
		if allTrain == 1 then 
			local info = t[idx]:split(",")
			xPath = "../train/"..info[2] .. "/" .. info[3]
		else 
			xPath = "../"..t[idx]
		end
		x = image.loadJPG(xPath, 3)
		x = loadData.rescale(x,params.inW,params.inH)
		x:resize(1,x:size(1),x:size(2),x:size(3))
		if testIdx == #t then testIdx = 1; finished = true else testIdx = testIdx + 1; finished = false end
		collectgarbage()
		return x:cuda(), xPath, finished 
	end
	x:resize(1,x:size(1),x:size(2),x:size(3))
	y:resize(1,1,y:size(1),y:size(2))
	collectgarbage()
	return x:cuda(),y:cuda()
end


function loadData.example()
	require "xlua"
	require "image"
	require "cunn"
	dofile("display.lua")
	params = {}
	params.inW = 128 
	params.inH = 128 
	params.outW = 32 
	params.outH = 32 
	params.batchSize = 1
	params.zoom = 4
	trainPaths, testPaths = loadData.init(1,3,1)

	for i =1, #makeMask do
		params.display = 1
		trainIdx = 1
		x,y = loadData.getData("train")
		--display(x,y,y,"train",imgPath)
	end

end


return loadData
