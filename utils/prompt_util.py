import random

def generate_human_prompt():
    races = ["An Asian", "An African", "A Caucasian", "A Mixed-race"]
    genders = ["man", "woman"]
    hair_colors = ["black", "brown", "blonde", "red", "gray", "white"]
    glasses = ["wearing glasses", "no glasses"]
    cloth_colors = [
        "red",
        "blue",
        "green",
        "black",
        "white",
        "yellow",
        "purple",
        "pink",
        "orange",
        "gray",
        "brown",
    ]
    tops = [
        "t-shirt",
        "shirt",
        "jacket",
        "sweater",
        "hoodie",
        "coat",
        "dress",
        "blouse",
    ]
    pants = ["jeans", "trousers", "shorts", "leggings"]
    shoes = ["sneakers", "boots", "sandals"]

    prompt = (
        f"{random.choice(races)} {random.choice(genders)} with {random.choice(hair_colors)} hair, "
        f"{random.choice(glasses)}, wearing a {random.choice(cloth_colors)} {random.choice(tops)}, "
        f"{random.choice(cloth_colors)} {random.choice(pants)}, and {random.choice(cloth_colors)} {random.choice(shoes)}"
    )

    return prompt