class Config:
    model = "resnet18"
    weight = "attribute_identifier/face_parsing/weights/resnet18.pt"
    input = "data/temporary_enlarged_face.png"
    output = "data/temporary_segmented"