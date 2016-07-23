local function fitSave(x,name,save,displayProb)
	--model:evaluate()
	local yPred = model:forward(x)
	if params.display == 1 and torch.uniform() < 0.001 then display(x,0,yPred,"test") end
	if save == 1 then
		local maskName = name:gsub(".jpg","_mask.jpg")
		image.saveJPG(maskName,yPred:squeeze())
	end
end


function slaveAllTest()
		slaves:addjob(function()
				       x, name, finished = loadData.getData("test",testPaths,0)
				       if finished == true then
					       print(string.format("Thread %d finished going to sleep",tid))
					       while true do sys.sleep(10) end
				       end
				       return x, name, finished
			       end,
			       function(x, name, finished)
					fitSave(x,name,save,0.01)
				end
				)
end

function slaveAllTrain()
		slaves:addjob(function()
				       x, name, finished = loadData.getData("test",allTrainPaths,1)
				       if finished == true then
					       print(string.format("Thread %d finished going to sleep",tid))
					       while true do sys.sleep(10) end
				       end
				       return x, name, finished
			       end,
			       function(x, name, finished)
					fitSave(x,name,save,0.01)
				end
				)
end

