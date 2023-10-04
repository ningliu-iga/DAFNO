# Domain Agnostic Fourier Neural Operators (DAFNO)
This repository houses the code for the following paper:
- [Domain Agnostic Fourier Neural Operators](https://arxiv.org/abs/2305.00478)

**Abstract**: Fourier neural operators (FNOs) can learn highly nonlinear mappings between function spaces, and have recently become a popular tool for learning responses of complex physical systems. However, to achieve good accuracy and efficiency, FNOs rely on the Fast Fourier transform (FFT), which is restricted to modeling problems on rectangular domains. To lift such a restriction and permit FFT on irregular geometries as well as topology changes, we introduce domain agnostic Fourier neural operator (DAFNO), a novel neural operator architecture for learning surrogates with irregular geometries and evolving domains. The key idea is to incorporate a smoothed characteristic function in the integral layer architecture of FNOs, and leverage FFT to achieve rapid computations, in such a way that the geometric information is explicitly encoded in the architecture. In our empirical evaluation, DAFNO has achieved state-of-the-art accuracy as compared to baseline neural operator models on two benchmark datasets of material modeling and airfoil simulation. To further demonstrate the capability and generalizability of DAFNO in handling complex domains with topology changes, we consider a brittle material fracture evolution problem. With only one training crack simulation sample, DAFNO has achieved generalizability to unseen loading scenarios and substantially different crack patterns from the trained scenario.

## Requirements
- [PyTorch](https://pytorch.org/)


## Running experiments
To run the elasticity example in the DAFNO paper
```
python3 elas_eDAFNO.py
python3 elas_iDAFNO.py
```

## Datasets
We provide the elasticity dataset that is used in the paper.

- [Elasticity dataset](https://drive.google.com/drive/folders/1ounVgMFcMO-iSR2111Xf4YfNddd2ialv?usp=sharing)
- [Original datasets from the Geo-FNO paper (elasticity, airfoil, and more)](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8)

## Citation

```
@inproceedings{liu2023dafno,
  title={Domain Agnostic Fourier Neural Operators},
  author={Liu, Ning and Jafarzadeh, Siavash and Yu, Yue},
  booktitle={Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS 2023)}
}
```
