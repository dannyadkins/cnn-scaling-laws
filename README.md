This is the implementation of the exercise for "Scaling Laws" from Jacob Hilton's Deep Learning Curriculum: https://github.com/jacobhilton/deep_learning_curriculum/blob/master/2-Scaling-Laws.md

# Findings 
Using a convnet with 2 conv layers, 1 pooling layer, and 1 linear layer, we got loss down .

Oddly... loss started to go up when the model got really big (e.g. 512 width).