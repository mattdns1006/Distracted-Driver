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
local nFeatsInc = torch.floor(params.nFeats/4)
local nOutputs
local nInputs
local kS = params.kernelSize
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
	for i =1,params.nDown do
		if i == 1 then nInputs = 3; else nInputs = nOutputs; end
		if i == 1 then nOutputs = nFeats; else nOutputs = nOutputs + nFeatsInc; end
		model:add(basicblock(nInputs,nOutputs,1))
		model:add(Pool(2,2,2,2,0,0))
		model:add(Dropout(0.2))
	end
	for i=1, params.nUp do
		nInputs = nOutputs
		nOutputs = nOutputs - nFeatsInc
		model:add(basicblock(nInputs,nOutputs,1))
		model:add(Dropout(0.2))
		model:add(UpSample(2))
	end
	nInputs = nOutputs
	model:add(Convolution(nInputs,1,3,3,1,1,1,1))
	model:add(nn.Sigmoid())
	layers.init(model)
	return model
end

function models.model3()
	
	function block(model,nInputs,nOutputs)
		model:add(Convolution(nInputs,nOutputs,3,3,1,1,1,1))
		model:add(SBN(nOutputs))
		model:add(af())

	end

	local model = nn.Sequential()
	local nInputs = 16
	local nOutputs = 16

	-- Down
	for i =1, 6 do
		if i == 1 then nInputs = 1; else nInputs = nOutputs; end
		block(model,nInputs,nOutputs)
		model:add(Pool(3,3,2,2,1,1))
	end

	x1 = nn.Identity()()

	layers.init(model)
	return model
end

return models
