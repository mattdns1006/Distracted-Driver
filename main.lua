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
cmd:option("-modelName","model1.model","Name of model.")
cmd:option("-modelSave",5000,"How often to save.")
cmd:option("-loadModel",0,"Load model.")
cmd:option("-nThreads",10,"Number of threads.")
cmd:option("-cv",1,"Cross validation.")
cmd:option("-batchSize",5,"Batch size duh.")

cmd:option("-inW",128,"Input size")
cmd:option("-inH",128,"Input size")

cmd:option("-sf",0.7,"Scaling factor.")
cmd:option("-nFeats",16,"Number of features.")
cmd:option("-level",0,"Which level (downsample).")
cmd:option("-nDown",8,"Which level (downsample).")

cmd:option("-lr",0.0001,"Learning rate.")
cmd:option("-lrDecay",1.2,"Learning rate change factor.")
cmd:option("-lrChange",10000,"How often to change lr.")
cmd:option("-weightDecay",0.005,"Regularization term.")


cmd:option("-display",0,"Display images.")
cmd:option("-displayFreq",100,"Display images frequency.")
cmd:option("-zoom",3,"Image zoom.")
cmd:option("-run",1,"Run.")
cmd:text()
dofile("donkeys.lua")
dofile("provider.lua")
dofile("/home/msmith/torchFunctions/counter.lua")

params = cmd:parse(arg)

optimMethod = optim.adam
models = require("models")
if params.loadModel == 1 then
	print("==> Loading model")
	model = torch.load(params.modelName):cuda()
else 	
	model = models.model2():cuda()
end
print(model)
criterion = nn.CrossEntropyCriterion():cuda()

function run()
	classes = {1,2,3,4,5,6,7,8,9,10}
	cmTrain = optim.ConfusionMatrix(#classes,classes)
	cmTest = optim.ConfusionMatrix(#classes,classes)
	epoch = 1

		newEpoch()
	while epoch < 20 do

		print("On epoch ==>",epoch)
		print("Training ==>")
		model:training()
		train()
		print("Testing ==>")
		model:evaluate()
		test()
		epoch = epoch + 1

		collectgarbage()
		print("Saving model ==>")
		torch.save(params.modelName,model)
	end
	--donkeys:terminate()
end

if params.run == 1 then run() end
	










