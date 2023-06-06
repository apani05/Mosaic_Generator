## Title: Generating Mosaics: A recreation of the “Novel Artificial Mosaic Generation Technique”[1] in C++ with OpenCV

### *Names: Aishwarya Pani and Tabitha Roemish*

### Plan: 
We plan to recreate the pseudo code lined out in the paper in C++ using what we can from the OpenCV library and writing the rest. We want to add a trackbar for the tile size. At this time we do not have extensions to propose because recreating the algorithm appears to be challenging but we will reevaluate and propose an extension if possible.

### Input and Output: 
The program will input an image and output that image with a mosaic transformation. 

### Evaluation: 
To evaluate the project we will use the same pictures used in the paper and try to run them through our algorithm to create our own mosaics. We can then compare the results visually. 

### Github link: 
apani05/Mosaic_Generator (github.com)__
Final Presentation Link: https://youtu.be/R7m6wKitEjo

### Schedule:
1. May 1, 2023 - Proposal submission.
2. May 8, 2023 - Outline code in github repository and have determined what functions can be used from OpenCV library and what algorithms we will have to write. 
3. May 15, 2023 - Design submission week. 
Create outline for project paper. 
Have code outlined with what functions we will use, what will be their input and output. 
We will complete till non-maximum suppression on the scalar field, i.e. 40% of the code completely finished.
4. May 22, 2023 - 
At least 40% of the paper will be completed with respect to introduction and background. 
75% of the code will be finished.
5. May 29, 2023 - 
75% of the paper will be completed which includes some results, bibliography. 
95% of code complete.
Preliminary slides for presentation completed. 
6. Jun 5, 2023 - Final code / Project demonstration
7. Jun 7, 2023 - Code submission

### Any Code we will use or have found: 
We are planning to implement the pseudo code from scratch but will use any code that applies in OpenCV and use code we find for reference or as a cited source if it is line for line. The following sites may be useful for implementing the code. 

The paper uses Gradient Vector Flow to determine at what angle the tiles should be placed and there is some code for the GVF algorithm:
*Active Contours, Deformable Models, and Gradient Vector Flow (jhu.edu) [2] - there’s c code for the GVF algorithm. It has been added as a branch in github as reference: Mosaic_Generator/GVF.c at TabithaRoemish-Reference · apani05/Mosaic_Generator (github.com)*

The project paper also uses Robert’s Cross operator for edge detection and we were not able to find an equivalent in OpenCV, probably because it is sensitive to noise but as the paper called out, produces more visually appealing results:
*Python OpenCV - Roberts Edge Detection - GeeksforGeeks [3] - This python code shows how to implement the cross operator for edge detection so we could follow this general code in C++.* 


### Bibliography:

[1]  Battiato, Sebastiano & Blasi, Gianpiero & Gallo, Giovanni & Guarnera, Giuseppe & Puglisi, Giovanni. (2008). A Novel Artificial Mosaic Generation Technique Driven by Local Gradient Analysis. 5102. 76-85. 10.1007/978-3-540-69387-1_9. 

[2] Xu, Chenyang, and Jerry L. Prince. “Active Contours, Deformable Models, and Gradient Vector Flow.” iacl.ece.jhu.edu, iacl.ece.jhu.edu/Projects/gvf.

[3]  Bhavyajain4641. “Python OpenCV  Roberts Edge Detection.” GeeksforGeeks, Jan. 2023, www.geeksforgeeks.org/python-opencv-roberts-edge-detection.
