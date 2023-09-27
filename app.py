import gradio as gr
import glob
import os
import cv2
import glob
from models.model import Model
#from evaluation import evaluation
datasets = ["HRF","DR-HAGIS","OVRS","ISBI2016_ISIC", "IOSTAR","CHASEDB","KEVASIR-SEG"]
models = ["UNET","SA-UNET", "ATT-UNET","SAM", "SAM-ZERO-SHOT"]




def default_mask():
    return None
def default_metrics():
    return {}
def on_select_gallery(dataset: gr.SelectData):
    path = glob.glob(os.path.join("datasets",dataset.value[0],"test/images/*"))
    return gr.update(visible=True, value = path)
def on_select_dataset_path(dataset: gr.SelectData):
    path = glob.glob(os.path.join("datasets",dataset.value[0],"test/images/*"))
    return  path
def on_select_gallery_label(dataset: gr.SelectData):
    path =dataset.value[0]
    return gr.update(visible=True, value = "## "+path+" Dataset")
def on_select_gallery_set_mask_path(value: gr.SelectData,path,dataset):
    image_path = path[int(value.index)]
    return os.path.join("datasets", dataset,"test","masks",os.path.basename(image_path))
def set_index(value: gr.SelectData):
    return str(value.index)
def set_value(value: gr.SelectData):
    return str(value.value)[2:-2]
def segment(dataset, model, image_index, path):
    return gr.update(visible= True)
def segment_(dataset, model, image_index,path):
    image_path = path[int(image_index)]
    mask_path = os.path.join("datasets", dataset,"test","masks",os.path.basename(image_path))
    return model.predict(image_path, mask_path)
def on_select_model(model: gr.SelectData,dataset):
    return Model(model.value, dataset)

def update_model_dataset(dataset : gr.SelectData , model):
    return gr.update(value = model.set_dataset(dataset.value[0]))
def update_model_name(model_name : gr.SelectData, model):
    return gr.update(value = model.set_model(model_name.value))
def update_dropdown_labels(dataset : gr.SelectData):
    choises = glob.glob(f"models/models_checkpoints/{dataset.value[0]}/*pth")
    choises = list(map(lambda value: value.split("/")[-1].split('.')[0], choises))
    return gr.Dropdown.update(choices = choises, interactive=True, label = "Models")

def test(model):
    return gr.update(value=model.evaluation())



with gr.Blocks() as demo:
    ## States
    model_build = gr.State()
    mask_p_state  = gr.State()
    #models_state = gr.State([])
    dataset_state = gr.State("ISBI2016_ISIC")
    path = glob.glob(os.path.join("datasets",dataset_state.value,"test/images/*"))
    path = gr.State(value=path)
    model_state  = gr.State(value = Model("UNET", dataset_state.value))
    demo_header = gr.Markdown(value="# A Comparative Analysis of State-of-the-Art Deep learning Models for Medical Image")
    image_index = gr.Markdown(visible = False)
    mask_y_state = gr.State()

    dataset = gr.Dataset(label="Datasets",components = [gr.Textbox(visible= False)], samples = list(map(lambda item: [item], datasets)),)
    model = gr.Dropdown(interactive=False, label = "Models",value=lambda : "UNET")
    dataset.select(update_dropdown_labels,None,outputs = [model])
    model.select(update_model_name,[model_state] , [model_state])
    dataset_label = gr.Markdown()
    gallery = gr.Gallery(value = path.value,visible=True)
    dataset.select(on_select_gallery,None,outputs = [gallery])
    dataset.select(on_select_gallery_label,None,outputs = [dataset_label])
    dataset.select(set_value,None,outputs = [dataset_state])
    dataset.select(update_model_dataset,[model_state],outputs = [model_state])
    dataset.select(on_select_dataset_path,None,outputs = [path])
    segement_btn = gr.Button(value = "Segment")
    dataset_label = gr.Markdown(value="# Result")
    gallery.select(set_index, None,outputs= [image_index])
    #mask_y_image = gr.Image(scale=2)
    row = gr.Row(visible=False)
    segement_btn.click(segment, [dataset_state, model_state, image_index, path], [row])
    with row:
        mask_p_image = gr.Image(scale=2)
        mask_y_image = gr.Image(scale=2)
        metrics_label = gr.Label(label = "Metrics",inputs=[mask_p_image])
        mask_p_image.change(test, inputs=[model_state],outputs = [metrics_label])
    gallery.select(on_select_gallery_set_mask_path, [path, dataset_state],outputs= [mask_y_state])
    gallery.select(on_select_gallery_set_mask_path, [path, dataset_state],outputs= [mask_y_image])
    segement_btn.click(segment_, [dataset_state, model_state, image_index, path], [mask_p_image])
    dataset.select(default_mask,None, outputs = [mask_p_image])
    dataset.select(default_metrics, None,outputs= [metrics_label])
    gallery.select(default_metrics, None,outputs= [metrics_label])
    gallery.select(default_mask, None,outputs= [mask_p_image])
demo.launch()
