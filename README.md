# practicalmachinelearning
Repo for JHU Coursera Practical Machine Learning
https://github.com/cunningc/courses/tree/master/08_PracticalMachineLearning

using RPLOT instead of RATTLE
install.packages("rpart.plot")
rpart.plot(model$finalModel)

** PUSHING & VIEWING PROJECT IN GITHUB
**** Publishing updates to Github 
1. make updates to index.Rmd
2. Knit to HTML
3. Use Git tab in viewpane on the right to set Staged on index files
3. click Commit button to commit and push the updates to github

Note that it may take up to 15 minutes for the content you publish into your github repository to become visible on the public internet via the github.io domain. 
Open a web browser and navigate to
http://cunningc.github.io/practicalmachinelearning
to view the HTML version of your R Markdown file.
Note that a few students have reported difficulties accessing the index.html file without including the file name in the URL, as http://cunningc.github.io/repositoryname/index.html. If accessing the file by repository name alone does not work, try accessing it with the fully specified file name.

As you begin grading assignments, you will likely run into the situation where a student has not submitted a gh-pages link, and therefore you'll need to read the student's HTML file.
Use the URL http://htmlpreview.github.io/ to convert a raw html file into the interpreted version, including any charts / graphics that were generated by knitr.

