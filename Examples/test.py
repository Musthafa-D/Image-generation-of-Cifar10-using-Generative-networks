explore_X,explore_y = [],[]
for i in range(len(train_loader)):
    
    X,y = iter(train_loader).next()    
    explore_X.append(X.numpy())
    explore_y.append(y.numpy())
    
explore_X,explore_y = np.array(explore_X).reshape([60000,1,28,28]),np.array(explore_y).reshape([1875*32])
print(explore_X.shape,explore_y.shape)

def get_class_arrays(class_index):
    return explore_X[explore_y==class_index],explore_y[explore_y==class_index]

def get_mean_images():
    for i in range(10):
        class_arrayX,class_arrayY = get_class_arrays(i)
        mean_image = class_arrayX.mean(axis=0)
        plt.subplot(2,5,i+1)
        plt.axis("off")
        plt.title(i)
        plt.imshow(mean_image.squeeze())
        
get_mean_images()



from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch

# Assuming you've trained a GAN and have a generator model called 'generator'
# Let's first generate a bunch of samples from the latent space

n_samples = 5000  # number of samples to generate
latent_dim = 100  # dimension of the latent space, change as per your GAN design

# Generate random noise vectors
z = torch.randn(n_samples, latent_dim).to(device)  # 'device' should be your model's device

# Pass the noise vectors through the generator to get fake images
with torch.no_grad():
    fake_images = generator(z).cpu().numpy()

# Flatten the images for t-SNE
fake_images = fake_images.reshape(n_samples, -1)

# Use t-SNE to reduce the dimensionality of the data
tsne = TSNE(n_components=2, random_state=42)
fake_images_2D = tsne.fit_transform(fake_images)

# Now let's plot the results
plt.scatter(fake_images_2D[:, 0], fake_images_2D[:, 1])
plt.show()
