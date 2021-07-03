import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


METRICS = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.AUC(name="auc"),
    tf.keras.metrics.AUC(name="prc", curve="PR"),
]


def build_model(input_shape=[768]):
    """A simple Deep learning Model in Keras"""

    model = Sequential(
        [
            Dense(64, activation="relu", input_shape=input_shape),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid", name="output_prob"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=METRICS)

    return model


@tf.function(
    input_signature=[
        {
            "array": tf.TensorSpec([None, 768], dtype=tf.float32, name="array"),
            "key": tf.TensorSpec([None], dtype=tf.int32, name="key"),
        }
    ]
)
def keyed_predictions(input):
    features = input.copy()
    key = features.pop("key")
    output = model(features)

    return {"key": key, "predictions": output}


@tf.function(
    input_signature=[
        {
            "array": tf.TensorSpec([None, 768], dtype=tf.float32, name="array"),
        }
    ]
)
def default_predictions(input):
    output = model(input)
    return {"predictions": output}


if __name__ == "__main__":
    model = build_model()

    sample_input = tf.random.uniform(shape=[1, 768])

    predictions = model.predict(sample_input)

    print(predictions.shape)

    model.save("saved_models/default/", overwrite=True)

    model.save(
        "saved_models/keyed/",
        signatures={
            "serving_default": default_predictions,
            "keyed_preditions": keyed_predictions,
        },
        overwrite=True,
    )

    # saved_model_cli show --dir saved_models/keyed/ --tag_set serve
    # saved_model_cli show --dir saved_models/keyed/ --tag_set serve --signature_def serving_default
    # saved_model_cli show --dir saved_models/keyed/ --tag_set serve --signature_def keyed_preditions
    # saved_model_cli show --dir saved_models/keyed/ --all

    loaded_model = tf.saved_model.load("saved_models/keyed/")
    print(list(loaded_model.signatures.keys()))

    infer_default = loaded_model.signatures["serving_default"]
    infer_keyed = loaded_model.signatures["keyed_preditions"]
    print(infer_keyed.structured_outputs)

    print("inferring:")
    labeling = infer_keyed(array=sample_input, key=tf.constant(1))
    print(labeling)
