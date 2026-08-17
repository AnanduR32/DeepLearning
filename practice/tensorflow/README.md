# Introduction

**Tensors** are the basic building blocks of tensorflow, it represents graphs that can produce it's own individual outputs, there can be numerous tensors that exists for a model, and when a session runs a series of tensors produces outputs up till the final result of the model.   

Tensors can have ranks, a rank 1 tensor represents a vector whereas a rank 2 and above represent matrices, but unlike those in numpy arrays these tensors can have varying datatypes. 

They can be of 4 types:
* Variable  
* Constant  
* Placeholder  
* SparseTensor  

And can be created as below: 
```python
rank1_tensor = tf.Variable(['My','Name','is','Anandu'], tf.string)
rank2_tensor = tf.Variable([['Anandu R','Software Engineer','SOTI'],['Aishwarya Michael','Software Engineer','Infosys']])

print(f'\nTensor 1: \n\tRank: {tf.rank(rank1_tensor)}\n\tShape: {rank1_tensor.shape}')
print(f'\nTensor 2: \n\tRank: {tf.rank(rank2_tensor)}\n\tShape: {rank2_tensor.shape}')
```  

      Tensor 1: 
        Rank: 1
        Shape: (4,)

      Tensor 2: 
        Rank: 2
        Shape: (2, 3)
        
Reshaping tensors 

```python
tf.reshape(rank2_tensor, (3,2))
```  

      <tf.Tensor: shape=(3, 2), dtype=string, numpy=
      array([[b'Anandu R', b'Software Engineer'],
           [b'SOTI', b'Aishwarya Michael'],
           [b'Software Engineer', b'Infosys']], dtype=object)>

