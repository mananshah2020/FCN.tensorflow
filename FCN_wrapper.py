import FCN
import configparser

config = configparser.ConfigParser()
config.read("settings.ini")


FCN.FLAGS.batch_size = config["FCN Settings"]['batch_size']
FCN.FLAGS.learning_rate = config["FCN Settings"]['learning_rate']
FCN.FLAGS.debug = config["FCN Settings"]['debug']
FCN.FLAGS.image_augmentation = config["FCN Settings"]['image_augmentation']
FCN.FLAGS.dropout = config["FCN Settings"]['dropout']
FCN.FLAGS.mode = config["FCN Settings"]['mode']
FCN.MAX_ITERATION = config["FCN Settings"]['max_iterations']
FCN.NUM_CLASSES = config["FCN Settings"]['num_classes']
FCN.IMAGE_WIDTH, FCN.IMAGE_HEIGHT = config["FCN Settings"]['image_size']
