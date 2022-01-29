# kNas
These days with fast development of deep learning, the need to design an appropriate network structure has become an issue.
In recent years, Neural Architecture Search (NAS) has shown a great success in design such networks and has automated and replaced
human experts with computer algorithms successfully. kNas is a program that has implemented NAS and uses evolutionary methods
to search for best network structure.


### Dependencies

* OS: Linux
* Python3.8 or later

### Installing


* Currently, the only support is for the \*nix-based systems
* Currently, the only support is for the execuatble files not library files


To install simply
```
git clone https://github.com/keagleV/knas.git
cd knas
echo  PATH=$PATH:$PWD >>  $HOME/.bashrc
cd dist
pip3 install knas-1.0.tar.gz
```
### Files descriptions

| Filename     |  Description
| ------------- |  --------------   
| `knas`        |   The main executing program
| `knasConfFile` |  Defines the methods to parse the configuration file 
| `knasDataset.py` |  Defines the methods to download/load the necessary datasets 
| `knasEA.py` |  Defines the evolutionary operations to be performed on NAS problem
| `knasLN.py` |  Defines the methods to create training model, convolutional and dense layers
| `knasLogging` |  Defines the methods to log messages to the console for the KNAS program
| `knasModel.py` |  Defines the methods to train a created model
| `knasModelLogging.py` |  Defines the methods to log messages to the console for the model training 
| `Samples/knas.conf`| A configuration file sample



### Executing program
To execute the program, first create a configuration file same as the one in the Samples directory. After creating the configration file, execute:

```
knas -f PATH_TO_YOUR_CONF_FILE
```

## Future work
Due to lack of time to learn more about PyTorch and its structure, EA methods use object creation instead of object modifiction in both
modification and creation phase. So, in order to do better, the object modification possibility rather than creating new, will be added.



## Help

For any help through using this family, you can use -h or --help command line option to get help about that specific program.
In the case of any ambiguity or software bug or any collaboration, feel free to send me an email at my email address.


## Authors

Contributors names and contact info

NAME: Kiarash Sedghi<br /> 
EMAIL: kiarash.sedghi99@gmail.com




## Version History

* 1.0
    * Initial Release

## License

This project is licensed under the [GNU Affero General Public License v3.0] License - see the LICENSE file for details

