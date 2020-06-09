from libscat import *

W1, W2 = 512, 784
img = Image.open('h2.jpg')
img_np = np.array(img)
img_tensor = Fv.to_tensor(img)[1:2, :, :][None]

#print(img_tensor.shape)
w, h = img_tensor.shape[2], img_tensor.shape[3]
img_tensor = torch.nn.functional.interpolate(img_tensor, (W1, W2))



