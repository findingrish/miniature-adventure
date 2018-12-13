#This container contains your model and any helper scripts specific to your model.
FROM tensorflow/tensorflow

ADD model.py /opt/model.py
RUN chmod +x /opt/model.py

ADD awsMachineData.csv /opt/awsMachineData.csv
RUN chmod +x /opt/awsMachineData.csv

ADD awsMachineTag.csv /opt/awsMachineTag.csv
RUN chmod +x /opt/awsMachineTag.csv

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install numpy
RUN pip3 install tensorflow

ENTRYPOINT ["/usr/bin/python3"]
CMD ["/opt/model.py"]
 