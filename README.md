# FinalPythonProject

Welcome to AI! Description: You can create a AI supervised learning model system that can analyze and evaluate the images and label and match the images with labels!

In that program there are 88 image data is given and there are 10 different classes/labels are defined.

Input Layers given as,

 model.add(keras.layers.Conv2D(32, (3,3), padding='same', activation= 'relu',input_shape=(28, 28,1)))
 
 Output Layers given as,
 
     model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    
For the archetecture of the code, program gets input from the user to create the hidden layers. These Hidden layers can be created as,
    
    Conventional 2D (filters, kernel size)
    Dense (units)
    Max Pooolin 2D (pool size)
    
After inputs taken from the user, program compiles, fits and reshapes the code.

Then evaluating process happens and results saved as csv file.
