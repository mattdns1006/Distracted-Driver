require "nn"
require "nngraph"

local Convolution = nn.SpatialConvolution
local Pool = nn.SpatialMaxPooling
local fmp = nn.SpatialFractionalMaxPooling
local UpSample = nn.SpatialUpSamplingNearest
local SBN = nn.SpatialBatchNormalization
local Dropout = nn.Dropout
local af = nn.ReLU
local Linear = nn.Linear
local Dropout = nn.Dropout
local layers = dofile("/home/msmith/torchFunctions/layers.lua")

models = {}

function initParamsEg()
	params = {}
	params.kernelSize = 3
	params.nFeats = 22
	params.nDown = 7
	params.nUp = 3 
	model = nn.Sequential()
end
--initParamsEg()

local nFeats = params.nFeats 
local nFeatsInc = params.nFeats
local nOutputs
local nInputs
local kS = 3 
local pad = torch.floor((kS-1)/2)

function shortcut(nInputPlane, nOutputPlane, stride)
	return nn.Sequential()
		:add(Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0))
		:add(SBN(nOutputPlane))
		--:add(nn.Identity())
end
	
function basicblock(nInputPlane, n, stride)
	local s = nn.Sequential()

	s:add(Convolution(nInputPlane,n,3,3,1,1,1,1))
	s:add(SBN(n))
	s:add(af())

	return nn.Sequential()
	 :add(nn.ConcatTable()
	    :add(s)
	    :add(shortcut(nInputPlane, n, stride)))
	 :add(nn.CAddTable(true))
	 :add(af())

end

function models.model1()
	local model = nn.Sequential()
	local nInputs
	local nOutputs
	for i =1, 8 do
		if i == 1 then nInputs = 3; else nInputs = nOutputs; end
		if i == 1 then nOutputs = nFeats; else nOutputs = nOutputs; end
		model:add(basicblock(nInputs,nOutputs,1))
		model:add(fmp(2,2,0.8,0.8))
		--model:add(Dropout(0.3))
	end
	model:add(Convolution(nInputs,1,3,3,1,1,1,1))
	nInputs = nOutputs
	outputBeforeReshape = model:cuda():forward(torch.rand(1,3,params.inH,params.inW):cuda()):size()
	nOutputsBeforeReshape = outputBeforeReshape[2]*outputBeforeReshape[3]*outputBeforeReshape[4]
	model:add(nn.Reshape(nOutputsBeforeReshape))
	model:add(nn.Linear(nOutputsBeforeReshape,100))
	model:add(nn.BatchNormalization(100))
	model:add(af())
	--model:add(Dropout(0.5))
	model:add(nn.Linear(100,10))
	layers.init(model)
	return model
end

function models.model2()
	local model = nn.Sequential()
	local nInputs
	local nOutputs
	for i =1, 8 do
		if i == 1 then nInputs = 3; else nInputs = nOutputs; end
		if i == 1 then nOutputs = nFeats; else nOutputs = nOutputs + nFeatsInc; end
		model:add(Convolution(nInputs,nOutputs,3,3,1,1,1,1))
		model:add(SBN(nOutputs))
		model:add(af())
		model:add(fmp(2,2,0.7,0.7))

	end
	model:add(Convolution(nOutputs,nOutputs,3,3,1,1,1,1))
	model:add(SBN(nOutputs))
	model:add(af())
	outputBeforeReshape = model:cuda():forward(torch.rand(1,3,params.inH,params.inW):cuda()):size()
	nOutputsBeforeReshape = outputBeforeReshape[2]*outputBeforeReshape[3]*outputBeforeReshape[4]
	model:add(nn.Reshape(nOutputsBeforeReshape))
	model:add(nn.Linear(nOutputsBeforeReshape,100))
	model:add(nn.BatchNormalization(100))
	model:add(af())
	--model:add(Dropout(0.5))
	model:add(nn.Linear(100,10))
	layers.init(model)
	return model
end

function models.vgg()

	local vgg = nn.Sequential()

	-- building block
	local function ConvBNReLU(nInputPlane, nOutputPlane)
	  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
	  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
	  vgg:add(nn.ReLU(true))
	  return vgg
	end

	-- Will use "ceil" MaxPooling because we want to save as much
	-- space as we can
	local MaxPooling = nn.SpatialMaxPooling

	ConvBNReLU(3,64):add(nn.Dropout(0.3))
	ConvBNReLU(64,64)
	vgg:add(MaxPooling(2,2,2,2):ceil())

	ConvBNReLU(64,128):add(nn.Dropout(0.4))
	ConvBNReLU(128,128)
	vgg:add(MaxPooling(2,2,2,2):ceil())

	ConvBNReLU(128,256):add(nn.Dropout(0.4))
	ConvBNReLU(256,256):add(nn.Dropout(0.4))
	ConvBNReLU(256,256)
	vgg:add(MaxPooling(2,2,2,2):ceil())

	ConvBNReLU(256,512):add(nn.Dropout(0.4))
	ConvBNReLU(512,512):add(nn.Dropout(0.4))
	ConvBNReLU(512,512)
	vgg:add(MaxPooling(2,2,2,2):ceil())

	ConvBNReLU(512,512):add(nn.Dropout(0.4))
	ConvBNReLU(512,512):add(nn.Dropout(0.4))
	ConvBNReLU(512,512)
	vgg:add(MaxPooling(2,2,2,2):ceil())

	ConvBNReLU(512,512):add(nn.Dropout(0.4))
	ConvBNReLU(512,512):add(nn.Dropout(0.4))
	ConvBNReLU(512,512)
	vgg:add(MaxPooling(2,2,2,2):ceil())

	ConvBNReLU(512,512):add(nn.Dropout(0.4))
	ConvBNReLU(512,512):add(nn.Dropout(0.4))
	ConvBNReLU(512,512)
	vgg:add(MaxPooling(2,2,2,2):ceil())
	vgg:add(nn.View(512))

	classifier = nn.Sequential()
	classifier:add(nn.Dropout(0.5))
	classifier:add(nn.Linear(512,512))
	classifier:add(nn.BatchNormalization(512))
	classifier:add(nn.ReLU(true))
	classifier:add(nn.Dropout(0.5))
	classifier:add(nn.Linear(512,10))
	vgg:add(classifier)

	-- initialization from MSR
	local function MSRinit(net)
	  local function init(name)
	    for k,v in pairs(net:findModules(name)) do
	      local n = v.kW*v.kH*v.nOutputPlane
	      v.weight:normal(0,math.sqrt(2/n))
	      v.bias:zero()
	    end
	  end
	  -- have to do for both backends
	  init'nn.SpatialConvolution'
	end

	MSRinit(vgg)
	return vgg
end

return models
