# Something
FROM conda/miniconda3
# Add sample application

EXPOSE 8000

# RUN wget https://repo.continuum.io/archive/Anaconda3-2020.07-Linux-x86_64.sh
# RUN bash Anaconda3-2020.07-Linux-x86_64.sh
RUN conda create --name conda_ec2 python=3.6
# RUN conda activate conda_ec2
RUN conda install -y -n conda_ec2 flask 
RUN conda install -y -n conda_ec2 -c conda-forge librosa 
RUN conda install -y -n conda_ec2 tensorflow=2

COPY application.py .
COPY Models.py .
COPY Plotter.py .
COPY Signal_Processing.py .
COPY ssing_cnn_v1.h5 .
ADD static static
ADD templates templates
ADD uploads uploads
# Run it# ENTRYPOINT ["python", "/tmp/application.py"]
#ENTRYPOINT ["python","-u","application.py"]
ENTRYPOINT ["conda","run", "-v", "-n","conda_ec2" ,"python","-u","application.py"]
#ENTRYPOINT ["conda", "run", "-n", "conda_ec2", "python", "application.py"]
