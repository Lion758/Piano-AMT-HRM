## conda environment setup
project repo should already be pulled so that environment.yml file is in root

* install miniconda / conda After installation add these to your environment variables then PATH

C:\Users\YOUR USERNAME\miniconda3

C:\Users\<YOUR USERNAME>\miniconda3\Library\bin

C:\Users\YOUR USERNAM\miniconda3\Scripts

**after here you may need to restart**


* In the project folder terminal run these to create and activate the environment 

    conda env create -f environment.yml

    conda activate piano_amt

**NOTE if using vs code windows & powershell does not work for conda try git bash terminal**

to test if environment is setup in the terminal run  ;

python --version (you should see 3.10.19)

<hr>

after then run 

    pip install -r requirementstest.txt

To get spleeter run 

    pip install spleeter


to run spleeter run 

    pleeter separate -o audio_output -p spleeter:5stems turnthelightsbackon.mp3