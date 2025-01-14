from T_Eva.ItSPT import ItSPT
pipeline = ItSPT(
    dataset="hotel-waimai",
    div=100,
    itera=10
)
pipeline.process_1()
pipeline.process_2()