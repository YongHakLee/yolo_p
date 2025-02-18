from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x.pt")

""" Extra Freeze Options
"1":
    Freeze all the layers from model.0 to model.23.cv3.1
        except [model.23.cv3.0.2, model.23.cv3.1.2]
    Freeze layers [model.23.cv3.2.0.0, model.23.cv3.2.0.1,
                    model.23.cv3.2.1.0, model.23.cv3.2.1.1]
    Train layers [dfl + model.23.cv3.2.2 + model.23.cv3.1.2 + model.23.cv3.0.2]

"2":
    Freeze all the layers from model.0 to model.23.cv3.1
        except [model.23.cv3.0.2, model.23.cv3.1.2]
    Freeze layers [model.23.cv3.2.0.0, model.23.cv3.2.0.1, model.23.cv3.2.1.0]
    Train layers [dfl + model.23.cv3.2.2 + model.23.cv3.2.1.1
                    + model.23.cv3.1.2 + model.23.cv3.0.2]

"3":
    Freeze all the layers from model.0 to model.23.cv3.1
        except [model.23.cv3.0.2, model.23.cv3.1.2]
    Freeze layers [model.23.cv3.2.0.0, model.23.cv3.2.0.1]
    Train layers [dfl + model.23.cv3.2.2 + model.23.cv3.2.1.1
                    + model.23.cv3.2.1.0 + model.23.cv3.1.2 + model.23.cv3.0.2]

"4":
    Freeze all the layers from model.0 to model.23.cv3.1
        except [model.23.cv3.0.2, model.23.cv3.1.2]
    Freeze layers [model.23.cv3.2.0.0]
    Train layers [dfl + model.23.cv3.2.2 + model.23.cv3.2.1.1
                    + model.23.cv3.2.1.0 + model.23.cv3.2.0.1
                    + model.23.cv3.1.2 + model.23.cv3.0.2]

"5":
    Freeze all the layers from model.0 to model.23.cv3.1
        except [model.23.cv3.0.2, model.23.cv3.1.2]
    Train layers [dfl + model.23.cv3.2.2 + model.23.cv3.2.1.1 + model.23.cv3.2.1.0
                    + model.23.cv3.2.0.1 + model.23.cv3.2.0.0
                    + model.23.cv3.1.2 + model.23.cv3.0.2]
"""

# Train the model
train_results = model.train(
    data="datasets/ArTaxOr_test/data.yaml",  # path to dataset YAML
    epochs=1000,  # number of training epochs
    imgsz=1280,  # training image size
    device=[0, 1],  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu,
    freeze="3",  # Freezes the first N layers of the model or specified layers by index,
    # reducing the number of trainable parameters. Extra Freeze Options are available.
    batch=32,  # batch size
    project="runs/detect/ArTaxOr_test",  # project name
    name="1280_32_dfl+3.2.2+3.2.1.1+3.2.1.0",  # name of the experiment
)


train_results2 = model.train(
    data="datasets/ArTaxOr_test/data.yaml",  # path to dataset YAML
    epochs=1000,  # number of training epochs
    imgsz=1280,  # training image size
    device=[0, 1],  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu,
    freeze="4",  # Freezes the first N layers of the model or specified layers by index,
    # reducing the number of trainable parameters. Extra Freeze Options are available.
    batch=32,  # batch size
    project="runs/detect/ArTaxOr_test",  # project name
    name="1280_32_dfl+3.2.2+3.2.1.1+3.2.1.0+3.2.0.1",  # name of the experiment
)


train_results3 = model.train(
    data="datasets/ArTaxOr_test/data.yaml",  # path to dataset YAML
    epochs=1000,  # number of training epochs
    imgsz=1280,  # training image size
    device=[0, 1],  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu,
    freeze="5",  # Freezes the first N layers of the model or specified layers by index,
    # reducing the number of trainable parameters. Extra Freeze Options are available.
    batch=32,  # batch size
    project="runs/detect/ArTaxOr_test",  # project name
    name="1280_32_dfl+3.2.2+3.2.1.1+3.2.1.0+3.2.0.1+3.2.0.0",  # name of the experiment
)
