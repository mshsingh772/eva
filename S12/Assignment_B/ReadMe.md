## Assignment B

### JSON File Description(coco format)
The coco format of the json file generated after annotating in using the VGG annotator looks like the one showed below: 

    "images": 
    [{
            "id": 0, #unique id assigned to each image
            "width": 1280, #width of the original image
            "height": 960, #height of the original image
            "file_name": "dog_image_1.jpg", #original image name
            "license": 1,
            "date_captured": ""  
    }],
    "annotation": 
    [{
            "id" : int, #unique id assigned to each bounding box irrespective of the image it belongs to
            "image_id": int, # maps the annotated region to the respective image so as to handle multiple regions in a single image
            "category_id": int,
            "segmentation": RLE or [polygon], # the type of the  segmentation used, like box, polygon circle etc.,
            "area": float, #area of the annotated region
            "bbox": [x,y,width,height], #the x,y coordinatea and width, height of the bbox
            "iscrowd": 0 or 1,
        }],
    "categories": 
    [{
            "id": int, # unique id to the category/attribute
            "name": str, # name of the category/attribute
    }]


### Elbow graph

![Image description](https://github.com/hemendra-06/EVA4/blob/master/S12/Assignment_B/images/Elbow_graph.PNG)


### Cluster without log conversion graph
![Image description](https://github.com/hemendra-06/EVA4/blob/master/S12/Assignment_B/images/Clusters_without_log.PNG)


### Cluster with log conversion graph
![Image description](https://github.com/hemendra-06/EVA4/blob/master/S12/Assignment_B/images/Clusters_with_log.PNG)