---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Vision Module Capstone

We will put our knowledge of neural networks and working with visual data to use by creating a program that detects and recognizes faces in a pictures, in order to sort the pictures based on individuals.
The goal is to 

1. take input from our camera
2. locate faces in the image
3. determine if there is a match for each face in the database
4. return the image with rectangles around the faces along with the corresponding name (or "Unknown" if there is no match)

In the "Unknown" case, the program should prompt the user to input the unknown person's name so they can be added to the database.
Here is an example of what might be returned if the program recognizes everyone in the image

<!-- #raw -->
<div style="text-align: center">
<p>
<img src="../_images/face_rec_example.png" alt="example face rec output" width=500>
</p>
</div>
<!-- #endraw -->

Let's take a closer look at the pre-trained models we'll be using to accomplish this.

## Pre-Trained FaceNet Models

We will utilize two models from pre-trained neural networks provided by `facenet_pytorch`.
The first is a model called `MTCNN`, which provides face *detection* capabilities. 
Given an input image, the model will return a list of box coordinates with corresponding probabilities for each detected face.
Let's develop some intuition about how this model works.
During training, images are broken into multiple boxes that are each "searched" for a face.
An issue with this method is that cross-entropy loss skews the results towards the empty boxes (boxes with no detected objects).
This is accounted for by weighting the boxes such that boxes with more content are weighted heavier.
The result is a model that can effectively identify multiple faces in an image.

Returning to how to use the trained model, you will be able to manipulate the box coordinates that are produced to create `Rectangle` objects that can be displayed around each face for the final product. 
You will also use these coordinates to crop your image such that it only displays the face you are attempting to recognize.
This cropped image can be passed into our next pre-trained model to produce a **descriptor vector** for the face (this will need to be done for each face detected in your image).

We will use `facenet_pytorch`'s `InceptionResnetV1`, which is trained to produce 512-dimensional face **descriptor vectors** for a given image of a face.
A facial descriptor vector is essentially an *embedding* of a face that numerically describes abstract *features*.
These features are not necessarily concrete facial features like a nose and eyes, but more abstract representations that the model learned.
During training, the model learned that these abstract features are effective at distinguishing between distinct faces and finding similarities between similar faces.
Thus, different images of the same face will have similar descriptor vectors whereas images of different faces will likely have drastically different descriptor vectors.
The model learned to create facial descriptors in this way by calculating loss and updating model parameters based on the similarity between descriptors of two of the same faces and two different faces.
Loss was calculated by creating descriptors of three face images - two of the same faces and one different - such that similarity between descriptors of the same face was encouraged and similarity between descriptors of different faces was discouraged (parameters were updated so that these descriptors were even more distinct).

The principle that images of the same face have similar descriptor vectors allows us to "recognize" a face after it has been detected.
If a detected face is "close enough" to a face in our database (the calculated distance between the face descriptors is below a certain cutoff), we can label the face with the appropriate name in the output image.
Otherwise, we can prompt the user to enter the name corresponding to the unknown face.

Now that we have some familiarity with the tools we'll be employing to accomplish facial recognition, let's talk about how our database can be structured to keep track of our faces and add new ones when we find them.

## Database

A useful way to structure a database is as a `dictionary` object, which allows us to easily point to a profile in the database using a *key*, like an individual's name.
Make sure you're familiar with Python's dictionary data structure, which can be reviewed [here](https://www.pythonlikeyoumeanit.com/Module2_EssentialsOfPython/DataStructures_II_Dictionaries.html) on PLYMI.
Another important tool to familiarize yourself with is the `pickle` module, which will allow you to store and load objects from your computer's file system.
PLYMI's coverage of the `pickle` module can be found [here](https://www.pythonlikeyoumeanit.com/Module5_OddsAndEnds/WorkingWithFiles.html#Saving-&-Loading-Python-Objects:-pickle).

Like we mentioned earlier, the database will be comprised of profiles that we point to using an individual's name.
These profiles will store information pertinent to facial recognition, including a name, a collection of face descriptor vectors, and a mean descriptor vector.
The collection of descriptor vectors will come from passing multiple images of the person through the models described in the previous section. 
The mean face descriptor will be used to compare to the faces we are trying to recognize.

By also writing functionality to add descriptor vectors and even entirely new profiles, you will be able to strengthen your database when a face is deemed "unknown" by adding the new image to the proper person or creating a new profile depending on whether or not the inputted name already exists in the database.

So we can 

* encode an image of a face using the our two FaceNet models to produce face descriptors
* organize these descriptors into meaningful profiles in a database of faces

But how do we actually recognize a face in a new image?
Let's take a look.

## Recognizing Faces

We mentioned earlier that the nature of face descriptor vectors is that images of the same face should have similar face descriptors.
Thus, in order to identify if a new image is a match to any of the faces in the database we must mathematically compute the similarity between the new face descriptor and each of the mean face descriptors in the database.
This can be done with **cosine distance**, which is a measure of the similarity between two normalized vectors.
cosine distance can be computed by taking the dot product of two normalized vectors.
Review ["Fundamentals of Linear Algebra"](https://rsokl.github.io/CogWeb/Math_Materials/LinearAlgebra.html#The-Dot-Product) for additional coverage on this topic.

We can use cosine distance to compute the similarity between any two face descriptors, but how similar is "close enough" to validate a match?
This is where a **cutoff** comes into play.
The cutoff indicates the maximum distance between two descriptors that is permitted to deem them a match. 
This value should be determined experimentally such that it is large enough to account for variability between descriptors of the same face but not so large as to falsely identify a face.
If a face descriptor doesn't fall below the cutoff distance with any face in the database, it is deemed "Unknown" and the user is prompted to enter a name.
If the name exists in the database, the image should be added to that person's profile.
This situation may arise from a bad photo (bad lighting, something covering the face, etc.) or too strict of a cutoff (in this case, experiment with a slightly larger cutoff).
If the name doesn't already exist, you should make a new profile with that name and face descriptor.

## Whispers Algorithm

The second part of this capstone project involves implementing an algorithm that can separate images into clusters of pictures of the same person.
There will be one cluster for each person in our database.
The implementation of this algorithm is explored in the following page.

## Team Tasks

This has been a basic run-through of the concepts and tools you will use to create this capstone project.
Here are some general tasks that it can be broken down into.

* Functionality to generate face descriptors using `MTCNN` and `InceptionResnetV1`
* Create a `Profile` class with functionality to add face descriptors and compute the mean descriptor
* Functionality to create, load, and save a database of profiles
    * Functionality to add and remove profiles
    * Funcitonality to add an image to the database, given a name (create a new profile if the name isn't in the database, otherwise add the image's face descriptor vector to the proper profile)
* Function to compute cosine distance between face descriptors
* Functionality to find a match for a face descriptor in the database, using a cutoff value
* Functionality to display an image with a box around detected faces with labels to indicate matches or an "Unknown" label otherwise
* Implement the whispers algorithm

## Links

* [Dictionary Data Structure - PLYMI](https://www.pythonlikeyoumeanit.com/Module2_EssentialsOfPython/DataStructures_II_Dictionaries.html)
* [Pickle Module - PLYMI](https://www.pythonlikeyoumeanit.com/Module5_OddsAndEnds/WorkingWithFiles.html#Saving-&-Loading-Python-Objects:-pickle)
* ["Fundamentals of Linear Algebra" - CogWeb](https://rsokl.github.io/CogWeb/Math_Materials/LinearAlgebra.html#The-Dot-Product) - **link needs to be changed when official website is published**
