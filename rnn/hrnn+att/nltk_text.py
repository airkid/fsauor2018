# from nltk.tokenize import sent_tokenize, word_tokenize
# doc="吼吼吼，萌死人的棒棒糖，中了大众点评的霸王餐，太可爱了。一直就好奇这个棒棒糖是怎么个东西，大众点评给了我这个土老冒一个见识的机会。看介绍棒棒糖是用德国糖做的，不会很甜，中间的照片是糯米的，能食用，真是太高端大气上档次了，还可以买蝴蝶结扎口，送人可以买礼盒。我是先打的卖家电话，加了微信，给卖家传的照片。等了几天，卖家就告诉我可以取货了，去大官屯那取的。虽然连卖家的面都没见到，但是还是谢谢卖家送我这么可爱的东西，太喜欢了，这哪舍得吃啊。"
# print(sent_tokenize(doc))



# import re
# # content = "去黄河路与文化路交叉口的建文新世界4楼奥斯卡看电影，在等电影进场的时候发现扶梯口的这家冰棒店，他家店里的冰棒各种各样的有十几种，都特别好看，老板推荐说冰棒都是进口奶源制作的，而且没有任何添加剂，抱着尝尝的态度，选了一根鲜果缤纷和一根芒果奇异果，吃的第一口就觉得棒棒的，奶味十足，水果吃起来也鲜，口感特别好，不像其它家的水果冰棒单纯把水果冻上去，吃起来特别硬特别冰，感觉这家还是比较有特色，种类比较多，价格也实在，老板服务也好，朋友说要把这家的冰棒尝一遍，给个好评"
# content = "quququ asdjasjd"
# print(re.split('(。|！|\!|？|\?|\r|\n|\r\n|\ )',content))

a=[22,30,20, 8,22,31, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
, 0, 0, 0, 0, 0, 0, 0,23,11,30,29,123,44, 1, 0, 0, 0, 0
, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,20,18,13,42
,14,20, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
, 0, 0, 0,95,76, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,19,36,91,25,13, 0, 0, 0
, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,27
, 1, 1,27, 3, 4,22,28,24, 1, 1,20, 1, 1,35,14,29, 1, 1
,11,68, 9, 1, 1, 1, 9, 1,19,11,13, 8,39, 8,16, 6,16,44
, 6, 6,15, 9,11, 4,21,11,14, 6, 0, 0, 0,31,29,11, 7,19
,15, 1, 1,16, 8, 1, 1, 5, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0
, 0, 0]
print(max(a))
