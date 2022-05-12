from argparse import ArgumentParser
from pathlib import Path

from tensorflow import keras

# Define this script's flags
parser = ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epochs', type=int, default=5)
parser.add_argument('--data_dir', type=str, default="./data/")
args = parser.parse_args()

# Make sure data_dir is absolute + create it if it doesn't exist
data_dir = Path(args.data_dir).absolute()
data_dir.mkdir(parents=True, exist_ok=True)

# Download and/or load data from disk
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(data_dir / 'mnist.npz')

# Standardize X's to be between 0.0-1.0 instead of 0-255
x_train, x_test = x_train.astype("float32") / 255, x_test.astype("float32") / 255

# Build Model
model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28, 1)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax'),
    ]
)

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=args.lr),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

# Train
history = model.fit(
    x_train,
    y_train,
    batch_size=args.batch_size,
    epochs=args.max_epochs,
    validation_split=0.1,
    callbacks=[keras.callbacks.TensorBoard(log_dir='./lightning_logs/keras')],
)

# Evaluate
model.evaluate(x_test, y_test)
