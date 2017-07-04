import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
from data import *
from clone import parameter


def show(subplace, title, _img):
    plt.subplot(*subplace)
    plt.axis('off')
    plt.title(title)
    plt.imshow(_img)
    plt.tight_layout()

def save_3_views(images, name, brightness=0, shadow=0):
    fig = plt.figure(figsize=(12, 20))
    titles = ["Center image", "Left image", "Right image"]
    for i in range(0, images.shape[0]):
        image = load_image(images[i])
        if (brightness):
            image = random_brightness(image)
        if (shadow):
            image = random_shadow(image)
        show((3, 1, i+1), titles[i], image)
    plt.tight_layout()
    savefig(parameter['saved_images_folder'] + name)

def save_flip_view(images, name):
    fig = plt.figure(figsize=(12, 8))
    image = load_image(images[0])
    show((2,1,1), "Original", image)
    image, _ = horizontal_flip(image, 0)
    show((2,1,2), "Flipped", image)
    plt.tight_layout()
    savefig(parameter['saved_images_folder'] + name)

def steering_angles_histogram(steering_angles, name, title, bins='auto', raw=0, fully_augmented=0):
    fig = plt.figure(figsize=(12, 8))
    if not raw:
        steering_angles = np.append(steering_angles, steering_angles * -1.)
        if fully_augmented:
            steering_angles = np.append(steering_angles, steering_angles + parameter['steering_bias'])
            steering_angles = np.append(steering_angles, steering_angles - parameter['steering_bias'])
    plt.hist(steering_angles, bins=bins)  # arguments are passed to np.histogram
    plt.title(title)
    savefig(parameter['saved_images_folder'] + name)

def test_image_shift(images, steering_angle, name):
    fig = plt.figure(figsize=(12, 8))
    image = load_image(images[0])   # Load center image
    show((2,1,1), "Original - steering_angle = " + str(steering_angle), image)
    image, steering_angle = random_shift(image, steering_angle)
    show((2,1,2), "Shifted - steering_angle = " + str(steering_angle), image)
    plt.tight_layout()
    savefig(parameter['saved_images_folder'] + name)

def get_random_image_id(image_paths):
    '''
    Returns a random number within the range of images available
    '''
    return int(np.random.uniform()*image_paths.shape[0])


if __name__ == "__main__":

    image_paths, steering_angles = load_paths_labels(parameter['training_images_folder'])

    if 0:
        i = get_random_image_id(image_paths)
        save_3_views(image_paths[i], '3views.png')
        i = get_random_image_id(image_paths)
        save_3_views(image_paths[i], '3bright.png', brightness=1)
        i = get_random_image_id(image_paths)
        save_3_views(image_paths[i], '3shadow.png', shadow=1)

        i = get_random_image_id(image_paths)
        save_flip_view(image_paths[i], "flip.png")

        steering_angles_histogram(steering_angles, 'raw_histo.png', "Histogram of the raw data.", raw=1)
        steering_angles_histogram(steering_angles, 'histo.png', "Histogram of the data when reversed horizontally.")
        steering_angles_histogram(steering_angles, 'full_histo.png', "Histogram of the data when using side cameras and a bias of " + str(parameter['steering_bias']), fully_augmented=1)

    i = get_random_image_id(image_paths)
    test_image_shift(image_paths[i], steering_angles[i], 'shift.png')

    if 1:
        gen = batch_generator(image_paths, steering_angles, parameter, True)
        _, steers = next(gen)
        for i in range(0, 13):
            _, steers_new = next(gen)
            steers = np.append(steers, steers_new)
        steering_angles_histogram(steers, 'gen_histo', 'Histogram of the generator data : ' + str(steers.shape[0]) + ' samples.', raw=1)



    print("End of test.")
