------------------------------
------------------------------
--- LOCATION/DECONVOLUTIONAL -
------------------------------
------------------------------
require "image"
require "gnuplot"
require "nn"
require "cunn"
require "xlua"
require "optim"
require "gnuplot"

cmd = torch.CmdLine()
cmd:text()
cmd:text("Options")
cmd:option("-modelName","deconv1.model","Name of model.")
cmd:option("-modelSave",20000,"How often to save.")
cmd:option("-loadModel",0,"Load model.")
cmd:option("-nThreads",10,"Number of threads.")

cmd:option("-cv",1,"Train on subset.")
cmd:option("-actualTest",0,"Acutal test predictions.")

cmd:option("-inW",128,"Input size")
cmd:option("-inH",128,"Input size")
cmd:option("-outW",32,"Input size")
cmd:option("-outH",32,"Input size")

cmd:option("-nFeats",32,"Number of features.")
cmd:option("-kernelSize",3,"Kernel Size.")
cmd:option("-nDown",5,"Up how far.")
cmd:option("-nUp",3,"Down how far.")

cmd:option("-lr",0.0001,"Learning rate.")
cmd:option("-lrDecay",1.2,"Learning rate change factor.")
cmd:option("-lrChange",10000,"How often to change lr.")

cmd:option("-display",0,"Display images.")
cmd:option("-displayFreq",100,"Display images frequency.")
cmd:option("-displayGraph",0,"Display graph of loss.")
cmd:option("-displayGraphFreq",500,"Display graph of loss.")
cmd:option("-nIter",2000000,"Number of iterations.")
cmd:option("-zoom",2,"Image zoom.")

cmd:option("-ma",500,"Moving average.")
cmd:option("-run",1,"Run.")

cmd:text()

params = cmd:parse(arg)

print("==> Init threads")
dofile("slaves.lua")
Loss = dofile("/home/msmith/torchFunctions/loss.lua")
dofile("train.lua")
dofile("test.lua")
dofile("display.lua")

models = require("models")
if params.loadModel == 1 then print("Loading model ==> "); model = torch.load("deconv.model"):cuda() ; else model = models.model1():cuda() end
criterion = nn.MSECriterion():cuda()

function run()

	trainLosses = Loss.new(params.ma)
	obs = 1

	while true do
		model:training()
		for i = 1, 200 do
			slaveTrain()
			obs = obs + 1
			if obs % params.modelSave == 0 then
				torch.save("deconv.model",model)
			end
		end
		model:evaluate()
		for i = 1, 1 do
			slaveAllTest()
		end

	end
end

function fitMasks(trainOrTest)
	local f
	model:evaluate()
	save = 1
	if trainOrTest == "train" then f = slaveAllTrain else f = slaveAllTest end
	while true do
		f()	
	end
end

if params.run == 1 then run() end
	










