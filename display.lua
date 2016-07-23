function display(x,trainOrTest)
		if imgDisplay == nil then 
			local zoom = params.zoom or 2
			local initPic = torch.rand(1,128,128)
			imgDisplay0 = image.display{image=initPic, zoom=zoom, offscreen=false}
			imgDisplay1 = image.display{image=initPic, zoom=zoom, offscreen=false}
			imgDisplay = 1 
		end
		local title
		if trainOrTest == "train" then
			title = "Train"
			image.display{image = x,  win = imgDisplay0, legend = title.. " x."}
		else	
			title = "Test"
			image.display{image = x,  win = imgDisplay1, legend = title .. " x."}
		end
end
