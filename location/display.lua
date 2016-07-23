function display(x,y,yPred,trainOrTest)
	if params.display == 1 then 
		if imgDisplay == nil then 
			local zoom = params.zoom 
			local initPic = torch.rand(1,3,128,128)
			imgDisplay0 = image.display{image=initPic, zoom=zoom, offscreen=false}
			imgDisplay1 = image.display{image=initPic, zoom=zoom, offscreen=false}
			imgDisplay2 = image.display{image=initPic, zoom=zoom, offscreen=false}
			imgDisplay3 = image.display{image=initPic, zoom=zoom, offscreen=false}
			imgDisplay4 = image.display{image=initPic, zoom=zoom, offscreen=false}
			imgDisplay = 1 
		end
		local title
		if trainOrTest == "train" then
			title = "Train"
			image.display{image = x,     win = imgDisplay0, legend = title.. " x."}
			image.display{image = y,     win = imgDisplay1, legend = title.. " y."}
			image.display{image = yPred, win = imgDisplay2, legend = title.. " pred."}
		else	
			title = "Test"
			image.display{image = x,     win = imgDisplay3, legend = title .. " x."}
			image.display{image = yPred,     win = imgDisplay4, legend = title.. " pred."}

		end
	end
end
