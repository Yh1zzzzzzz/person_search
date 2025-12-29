
==================== 开始测试 270M Model ====================
正在加载模型: ./T5_270M_Base ...
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
Loading weights: 100%|█| 911/911 [00:00<00:00, 4485.42it/s, Materializing param=model.encoder.vision_tower.visi
Ground Truth: The man has short, dark hair and wears khaki pants with an oversized grey hoodie. His black backpack hangs from one shoulder.


>>> 类别: A. 通用场景类
Prompt                              | Loss     | Generated Output
--------------------------------------------------------------------------------------------------------------
The following generation flags are not valid and may be ignored: ['top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
<start_of_image> A photo of         | 4.3194   |  a man who was arrested for allegedly stealing a bag from a supermarket.The man was arrested on suspicion of theft and was taken to the police station.The man was arrested on suspicion of theft and was taken to the police station.The
<start_of_image> An image of        | 4.2897   |  a man carrying a backpack.The man is wearing a black backpack.The man is wearing a black backpack.The man is wearing a black backpack.The man is wearing a black backpack.The man is wearing a black backpack
<start_of_image> A picture showing  | 4.2925   |  a man carrying a bag.A picture showing a man carrying a bag.A picture showing a man carrying a bag.A picture showing a man carrying a bag.A picture showing a man carrying a bag.A picture showing
<start_of_image> In this image, there is | 4.1656   |  a man wearing a black backpack.The man is wearing a black backpack.The man is wearing a black backpack.The man is wearing a black backpack.The man is wearing a black backpack.The man is wearing a black
<start_of_image> A view of          | 4.3967   |  the entrance to the airport.The airport is located in the city of Antalya, Turkey.The airport is located in the city of Antalya, Turkey.The airport is located in the city of Antalya, Turkey.The airport is located

>>> 类别: B. 人物主体类 (ReID 核心)
Prompt                              | Loss     | Generated Output
--------------------------------------------------------------------------------------------------------------
<start_of_image> The person is      | 4.4554   |  a 20-year-old male who is a resident of the 1000 block of 1000 block of 1000 block of 1000 block of 1000 block of
<start_of_image> The person is wearing | 4.4937   |  a backpack.The person is wearing a backpack.The person is wearing a backpack.The person is wearing a backpack.The person is wearing a backpack.The person is wearing a backpack.The person is wearing a backpack
<start_of_image> This person is carrying | 4.3901   |  a bag.This person is carrying a bag.This person is carrying a bag.This person is carrying a bag.This person is carrying a bag.This person is carrying a bag.This person is carrying a bag
<start_of_image> A pedestrian who is | 4.6613   |  wearing a backpack is struck by a car. The pedestrian is struck by a car. The pedestrian is struck by a car. The pedestrian is struck by a car. The pedestrian is struck by a car. The pedestrian is struck by a car. The
<start_of_image> The pedestrian is wearing | 4.7035   |  a backpack.The pedestrian is wearing a backpack.The pedestrian is wearing a backpack.The pedestrian is wearing a backpack.The pedestrian is wearing a backpack.The pedestrian is wearing a backpack.The pedestrian is wearing a backpack

>>> 类别: C. 结构化/元数据类
Prompt                              | Loss     | Generated Output
--------------------------------------------------------------------------------------------------------------
<start_of_image> Caption:           | 3.6786   |  A man is seen carrying a bag in the front of a bus.Photo by: J.A. H. S. J.
<start_of_image> Description:       | 3.7205   | * Age: 21* Gender: Male* Height: 5'11"* Weight: 110* Hair Color:
<start_of_image> Visual description: | 3.6677   | * Description:The 
<start_of_image> Attributes:        | 3.8151   | * Type: A* Type: A* Type: A* Type: A* Type:
<start_of_image> Summary:           | 3.8384   | * 1. 1. 1. 1. 1. 1. 1. 1. 1. 1

>>> 类别: D. 细粒度引导类
Prompt                              | Loss     | Generated Output
--------------------------------------------------------------------------------------------------------------
<start_of_image> The color of the upper clothing is | 4.6283   |  black.The color of the upper clothing is black.The color of the upper clothing is black.The color of the upper clothing is black.The color of the upper clothing is black.The color of the upper clothing is
<start_of_image> The person has     | 4.4223   |  been arrested for the following charges:* Attempted murder* Attempted robbery* Attempted burglary* Attempted theft* Theft of a motor vehicle* Theft of a vehicle* Theft of a motor vehicle* Theft
<start_of_image> Looking at the person's clothes, | 4.2604   |  I can see that he is wearing a black backpack.I can also see that he is wearing a black backpack.I can also see that he is wearing a black backpack.I can also see that he is wearing a black backpack.
<start_of_image> Upper body:        | 3.8912   |  100% cottonLower body: 100% cottonThe following is a list of the most common types of clothing that are used in the United States.* Cotton: Cotton is the most common type
<start_of_image> Full body shot of  | 4.1980   |  the suspect.The suspect is a 20-year-old male from the 1000 block of 1000 block of 1000 block of 1000 block of 100

==================== 开始测试 1B Model ====================
正在加载模型: ./T5_1B_model_Base ...
Loading weights: 100%|█| 1119/1119 [00:00<00:00, 4656.76it/s, Materializing param=model.encoder.vision_tower.vi
Ground Truth: The man has short, dark hair and wears khaki pants with an oversized grey hoodie. His black backpack hangs from one shoulder.


>>> 类别: A. 通用场景类
Prompt                              | Loss     | Generated Output
--------------------------------------------------------------------------------------------------------------
<start_of_image> A photo of         | 3.9568   |  a man who was seen leaving the airport on Wednesday afternoon.The man was seen leaving the airport on Wednesday afternoon.The man was wearing a grey hoodie, grey pants and black shoes.The man was carrying a black bag.The
<start_of_image> An image of        | 4.0712   |  a man wanted in connection with a robbery at a bank in the city of San Juan.The man is described as being of Asian origin, with a medium build, and wearing a black backpack.The man is wanted in connection with a robbery
<start_of_image> A picture showing  | 3.6953   |  a man who was seen leaving the airport on Saturday.The man was seen leaving the airport on Saturday afternoon.The man was wearing a grey hoodie, grey pants, and black shoes.The man was carrying a black bag.The
<start_of_image> In this image, there is | 3.6804   |  a man who is carrying a bag.The man is carrying a bag.The man is carrying a bag.The man is carrying a bag.The man is carrying a bag.The man is carrying a bag.The
<start_of_image> A view of          | 3.7339   |  the man who was arrested for allegedly stealing a backpack from a bus stop in the city of San Jose.The man was arrested for allegedly stealing a backpack from a bus stop in the city of San Jose.The man was arrested for allegedly stealing

>>> 类别: B. 人物主体类 (ReID 核心)
Prompt                              | Loss     | Generated Output
--------------------------------------------------------------------------------------------------------------
<start_of_image> The person is      | 3.9979   |  described as a white male, aged 20-30, with short black hair and a thin build. He was wearing a grey hooded top, grey jogging bottoms and black trainers.The person was last seen walking along the escalator in the
<start_of_image> The person is wearing | 4.0103   |  a grey hooded top, grey jogging bottoms and black shoes.The person is carrying a black bag.The person is described as being in their late teens to early twenties.The person is described as being of a slim build.The
<start_of_image> This person is carrying | 3.9549   |  a black bag.This person is carrying a black bag.This person is carrying a black bag.This person is carrying a black bag.This person is carrying a black bag.This person is carrying a black bag.
<start_of_image> A pedestrian who is | 4.0913   |  walking on the sidewalk in front of the entrance of the building is hit by a car. The pedestrian is taken to the hospital with serious injuries.The accident occurred at 10:30 a.m. on the 10th
<start_of_image> The pedestrian is wearing | 3.9412   |  a grey hooded top, dark pants and black shoes.The man is described as being in his early 20s, approximately 170cm tall and of a slim build.He was wearing a black backpack.Anyone with

>>> 类别: C. 结构化/元数据类
Prompt                              | Loss     | Generated Output
--------------------------------------------------------------------------------------------------------------
<start_of_image> Caption:           | 3.3933   |  
<start_of_image> Description:       | 3.3775   | Un homme de 20 à 30 ans, de 1m80 à 1m90, avec un poids de 80 à 90 kg, est apparu à la gare de Tokyo tôt ce matin.
<start_of_image> Visual description: | 3.2809   |  * 
<start_of_image> Attributes:        | 3.2711   | * Age: 20s* Gender: Male* Height: 170cm* Weight: 60kg* Hair: Black
<start_of_image> Summary:           | 3.3948   | A man was arrested after he was found with a large amount of drugs in his possession.The man was arrested on Tuesday, 11 August, at 10:00 am, at the airport.The man was found with

>>> 类别: D. 细粒度引导类
Prompt                              | Loss     | Generated Output
--------------------------------------------------------------------------------------------------------------
<start_of_image> The color of the upper clothing is | 3.9930   |  gray, and the pants are khaki. The shoes are black. The backpack is black.The suspect is a man in his 20s. He is about 170 cm tall and weighs about 60 kg. He has
<start_of_image> The person has     | 3.9800   |  been identified as 21-year-old Daniel M. C. H. S. S. 
<start_of_image> Looking at the person's clothes, | 3.7398   |  it is suspected that the person is a person who has been in the area for a long time.The person who was seen was a person who was seen in the area for a long time.The person who was seen was a person who
<start_of_image> Upper body:        | 3.2926   |  170 cmLower body: 65 cmWeight: 60 kgHair color: blackEye color: blackSkin color: whiteHair length: longHair style: straightHair color: black
<start_of_image> Full body shot of  | 3.7718   |  a man who was seen on the escalator at the airport. He was wearing a grey hoodie, grey pants and black shoes. He was carrying a black bag.
