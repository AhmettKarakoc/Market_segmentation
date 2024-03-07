FROM centos:centos8
#ADD gcp-key.json /gcp-key.json

RUN cd /etc/yum.repos.d/
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*
RUN sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*

RUN yum update -y
RUN yum install sudo -y
RUN yum groupinstall "Development Tools" -y
RUN yum install python39 -y
RUN yum install python3-pip -y
RUN pip3 install --upgrade pip

COPY src/ /market_segmentation/src
COPY data/ /market_segmentation/data
COPY models/ /market_segmentation/models
COPY ./requirements.txt /market_segmentation
#COPY ./pyproject.toml /market_segmentation
#COPY ./poetry.lock /market_segmentation
COPY ./ds_app.py /market_segmentation
#RUN mkdir /home/gcp_keys
#COPY gcp-key.json /home/gcp_keys/gcp-key.json

WORKDIR /market_segmentation

RUN pip3 install -r requirements.txt

#RUN pip install poetry
#RUN poetry config virtualenvs.create false
#RUN poetry install --no-dev
