# keras_model_aws_ec2
Example of deploying a keras model to an aws ec2 Instance

Following this tutorial : https://towardsdatascience.com/simple-way-to-deploy-machine-learning-models-to-cloud-fd58b771fdcf
tf.Keras vision model to classify cat vs dog

git pull https://github.com/CVxTz/keras_model_aws_ec2

cd keras_model_aws_ec2

sudo apt install docker.io

sudo docker build -t app-dog .

sudo docker run -p 80:80 app-dog .