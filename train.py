# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from config_directory import config
import matplotlib.pyplot as plt
from matplotlib.image import imread
from imutils import paths
import numpy as np
import random
import os

def plot_training(H, N, plot_path):
	
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plot_path)

def class_distribution(dataset_dir):

	# Create a list to hold the class names and their counts
	class_counts = []
	class_list = []

	# Go through each folder in the dataset directory
	for class_folder in os.listdir(dataset_dir):
		class_folder_path = os.path.join(dataset_dir, class_folder)
		class_list.append(class_folder)
		
		# Get the number of files in this folder
		num_images = len(os.listdir(class_folder_path))
		
		# Append the class name and its number of images to the list
		class_counts.append(num_images)

	return class_counts, class_list

def random_images_grid(dataset_dir, plot_path, grid_size):
	
	sample_images = []

	# Go through each folder in the dataset directory
	for class_folder in os.listdir(dataset_dir):
		class_folder_path = os.path.join(dataset_dir, class_folder)
		
		# Get the files in this folder
		folder_images = os.listdir(class_folder_path)
		sample_images.append([os.path.join(class_folder_path, image_path) for image_path in random.sample(folder_images, grid_size)])
	
	sample_images = [item for sublist in sample_images for item in sublist]
	for image_idx in range(1, grid_size*len(config.CLASSES)+1):
		plt.subplot(len(config.CLASSES), grid_size, image_idx)
		random_img = imread(sample_images[image_idx-1])
		plt.imshow(random_img)
		plt.savefig(plot_path)

def get_files_from_directory():
	# set directory path
	dir_path = "output"

	# get list of all files in directory
	files = os.listdir(dir_path)

	return files

def image_generators(train_augmentation, val_augmentation, img_size):
	# initialize the training generator
	train_generator = train_augmentation.flow_from_directory(
		train_path,
		class_mode="categorical",
		target_size=(img_size, img_size),
		color_mode=config.COLOR_MODE,
		shuffle=True,
		batch_size=config.BATCH_SIZE)

	# initialize the validation generator
	val_generator = val_augmentation.flow_from_directory(
		val_path,
		class_mode="categorical",
		target_size=(img_size, img_size),
		color_mode=config.COLOR_MODE,
		shuffle=False,
		batch_size=config.BATCH_SIZE)

	# initialize the testing generator
	test_generator = val_augmentation.flow_from_directory(
		test_path,
		class_mode="categorical",
		target_size=(img_size, img_size),
		color_mode=config.COLOR_MODE,
		shuffle=False,
		batch_size=config.BATCH_SIZE)
	
	return train_generator, val_generator, test_generator

def define_base_model(img_size):
	base_model = VGG19(weights="imagenet", include_top=False,
		input_tensor=Input(shape=(img_size, img_size, config.IMAGE_CHANNELS)))
	
	return base_model

# derive the paths to the training, validation, and testing
# directories
train_path = os.path.sep.join([config.BASE_PATH, config.TRAIN])
val_path = os.path.sep.join([config.BASE_PATH, config.VALIDATION])
test_path = os.path.sep.join([config.BASE_PATH, config.TEST])

train_dist, class_list = class_distribution(train_path)
val_dist, _ = class_distribution(val_path)
test_dist, _ = class_distribution(test_path)

# calculate the class distribution
class_dist = np.array(train_dist) + np.array(test_dist) + np.array(val_dist)

# Use matplotlib to create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(class_list, class_dist, color='skyblue')
plt.title('Number of Images in Each Class')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.savefig(config.CLASS_DISTRIBUTION)

# construct grids of the images from each class of the training and testing
# directories
random_images_grid(train_path, config.TRAIN_IMAGES_GRID, 5)

# determine the total number of image paths in training, validation,
# and testing directories
total_train = len(list(paths.list_images(train_path)))
total_val = len(list(paths.list_images(val_path)))
total_test = len(list(paths.list_images(test_path)))

# initialize the training data augmentation object
train_augmentation = ImageDataGenerator(rescale=1./255,
	rotation_range=30,
	# zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	# horizontal_flip=True,
	fill_mode="nearest")

# initialize the validation/testing data augmentation object (which
# we'll be adding mean subtraction to)
val_augmentation = ImageDataGenerator(rescale=1./255)

# # define the ImageNet mean subtraction (in RGB order) and set the
# # the mean subtraction value for each of the data augmentation
# # objects
# mean = np.array([123.68, 116.779, 103.939], dtype="float32")
# train_augmentation.mean = mean
# val_augmentation.mean = mean

for image_size in config.IMAGE_SIZE:
	# get the files from the directory where the model should be saved
	files = get_files_from_directory()

	if config.MODEL_NAME in files:
		print("MODEL ALREADY EXISTS")

		train_generator, val_generator, test_generator = image_generators(train_augmentation, val_augmentation, image_size)

		# load the saved model
		saved_model = load_model(config.MODEL_PATH)

		# create the base model
		base_model = define_base_model(image_size)
		
		# construct the head of the model that will be placed on top of
		# the base model
		head_model = base_model.output
		head_model = MaxPooling2D(pool_size=(2, 2))(head_model)
		head_model = Flatten(name="flatten")(head_model)
		head_model = Dense(512, activation="relu")(head_model)
		head_model = Dropout(0.5)(head_model)
		head_model = Dense(len(config.CLASSES), activation="softmax")(head_model)

		# place the head FC model on top of the base model (this will become
		# the actual model we will train)
		new_model = Model(inputs=base_model.input, outputs=head_model)
		new_model.summary()

		new_layers = len(new_model.layers) - len(saved_model.layers)
		
		# copy the weights from the saved model to the new model
		print("First block of base layers")
		for i, layer in enumerate(saved_model.layers[:len(base_model.layers)]):
			# print(new_model.layers[i])
			new_model.layers[i].set_weights(layer.get_weights())

		print("Second block of head layers")
		for i, layer in enumerate(saved_model.layers[len(base_model.layers):]):
			# print(new_model.layers[i+len(base_model.layers)+new_layers])
			new_model.layers[i+len(base_model.layers)+new_layers].set_weights(layer.get_weights())
			
		# compile new model
		print("[INFO] compiling model...")
		new_model.compile(loss="categorical_crossentropy", optimizer=SGD(learning_rate=1e-5, momentum=0.9),
			metrics=["accuracy"])

		# training the new model
		print("[INFO] training model using images with shape {:d} x {:d}...".format(image_size, image_size))
		H = new_model.fit(
			x=train_generator,
			steps_per_epoch=total_train // config.BATCH_SIZE,
			validation_data=val_generator,
			validation_steps=total_val // config.BATCH_SIZE,
			epochs=config.MODEL_EPOCHS)

		print("[INFO] evaluating model...")

		test_generator.reset()

		predictions = new_model.predict(x=test_generator,
			steps=(total_test // config.BATCH_SIZE))
		# print(predictions)

		predicted_classes = np.argmax(predictions, axis=1)
		# print(predicted_classes)

		# print(test_generator.classes)

		print(classification_report(test_generator.classes, predicted_classes,
			target_names=test_generator.class_indices.keys()))
		
		plot_path = config.UNFROZEN_PLOT_PATH
		plot_path = plot_path.split(".")
		plot_path = ''.join([plot_path[0], str(image_size), "x", str(image_size), ".", plot_path[1]])

		plot_training(H, config.MODEL_EPOCHS, plot_path)

		# reset the data generators
		train_generator.reset()
		val_generator.reset()

		# serialize the new model to disk
		print("[INFO] serializing network...")
		new_model.save(config.MODEL_PATH, save_format="h5")

	else:
		print("MISSING MODEL")

		train_generator, val_generator, test_generator = image_generators(train_augmentation, val_augmentation, image_size) 

		# load the VGG16 network, ensuring the head FC layer sets are left
		# off
		base_model = define_base_model(image_size)

		# construct the head of the model that will be placed on top of
		# the base model
		head_model = base_model.output
		head_model = Flatten(name="flatten")(head_model)
		head_model = Dense(512, activation="relu")(head_model)
		head_model = Dropout(0.5)(head_model)
		head_model = Dense(len(config.CLASSES), activation="softmax")(head_model)

		# place the head FC model on top of the base model (this will become
		# the actual model we will train)
		model = Model(inputs=base_model.input, outputs=head_model)
		model.summary()
		
		# loop over all layers in the base model and freeze them so they will
		# *not* be updated during the first training process
		for layer in model.layers[:len(base_model.layers)]:
			layer.trainable = False

		# loop over the layers in the model and show which ones are trainable
		# or not
		for layer in model.layers:
			print("{}: {}".format(layer, layer.trainable))
			
		# compile the model (this needs to be done after setting the
		# layers to being non-trainable)
		print("[INFO] compiling model...")
		model.compile(loss="categorical_crossentropy", optimizer=SGD(learning_rate=1e-5, momentum=0.9),
			metrics=["accuracy"])

		# train the head of the network for a few epochs (all other layers
		# are frozen) -- this will allow the new FC layers to start to become
		# initialized with actual "learned" values versus pure random
		print("[INFO] training head using images with shape {:d} x {:d}...".format(image_size, image_size))
		H = model.fit(
			x=train_generator,
			steps_per_epoch=total_train // config.BATCH_SIZE,
			validation_data=val_generator,
			validation_steps=total_val // config.BATCH_SIZE,
			epochs=config.HEAD_EPOCHS)

		# reset the testing generator and evaluate the network after
		# fine-tuning just the network head
		print("[INFO] evaluating after fine-tuning network head...")

		test_generator.reset()

		predictions = model.predict(x=test_generator,
			steps=(total_test // config.BATCH_SIZE))
		# print(predictions)

		predicted_classes = np.argmax(predictions, axis=1)
		# print(predicted_classes)

		# print(test_generator.classes)
		

		print(classification_report(test_generator.classes, predicted_classes,
			target_names=test_generator.class_indices.keys()))

		plot_training(H, config.HEAD_EPOCHS, config.WARMUP_PLOT_PATH)

		# reset the data generators
		train_generator.reset()
		val_generator.reset()

		# now that the head FC layers have been trained/initialized, lets
		# unfreeze the CONV layers and make them trainable
		for layer in model.layers[:len(base_model.layers)]:
			layer.trainable = True

		# loop over the layers in the model and show which ones are trainable
		# or not
		for layer in model.layers:
			print("{}: {}".format(layer, layer.trainable))
			
		# for the changes to the model to take affect we need to recompile
		# the model, this time using SGD with a *very* small learning rate
		print("[INFO] re-compiling model...")
		model.compile(loss="categorical_crossentropy", optimizer=SGD(learning_rate=1e-5, momentum=0.9),
			metrics=["accuracy"])

		# train the model again, this time fine-tuning *both* the
		# CONV layers along with the set of FC layers
		print("[INFO] training model using images with shape {:d} x {:d}...".format(image_size, image_size))
		H = model.fit(
			x=train_generator,
			steps_per_epoch=total_train // config.BATCH_SIZE,
			validation_data=val_generator,
			validation_steps=total_val // config.BATCH_SIZE,
			epochs=config.MODEL_EPOCHS)

		# reset the testing generator and then use the trained model to
		# make predictions on the data
		print("[INFO] evaluating after fine-tuning network...")

		test_generator.reset()

		predictions = model.predict(x=test_generator,
			steps=(total_test // config.BATCH_SIZE))
		# print(predictions)

		predicted_classes = np.argmax(predictions, axis=1)
		# print(predicted_classes)

		# print(test_generator.classes)

		print(classification_report(test_generator.classes, predicted_classes,
			target_names=test_generator.class_indices.keys()))

		plot_path = config.UNFROZEN_PLOT_PATH
		plot_path = plot_path.split(".")
		plot_path = ''.join([plot_path[0], str(image_size), "x", str(image_size), ".", plot_path[1]])

		plot_training(H, config.MODEL_EPOCHS, plot_path)

		# serialize the model to disk
		print("[INFO] serializing network...")
		model.save(config.MODEL_PATH, save_format="h5")
