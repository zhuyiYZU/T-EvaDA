from T_Eva.Spot import Spot

pipeline = Spot(
    model_name="bert",
    model_path="/home/ubuntu/Lizhenglong/T-Eva/T_Eva/prompt_base_model",
    dataset="hotel-waimai",
    template_id=0,
    verbalizer_type="manual",
    seed=42,
    max_epochs=3,
    batch_size=16
)
pipeline.load_model_and_data()
pipeline.train()
pipeline.test()
pipeline.test_new()