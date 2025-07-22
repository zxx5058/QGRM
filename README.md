<h1 align="center">
 Real-time Respiration Monitoring via Motion Artifact Suppression and Quality-Guided Peak Detecion (QGRM)
</h1>

<p align="center">
  <strong>Xinxin Zhang <sup>1</sup></sup></strong>
  .
  <strong>Gan Pei <sup>1</sup></sup></strong>
  .
  <strong>Chenrui Niu <sup>1</sup></sup></strong>
  .
  <strong>Feng Zheng<sup>1</sup></strong>
  .
  <strong>Guangtao Zhai<sup>2</sup></strong>
  .
  <strong>Xiao-Ping Zhang<sup>3</sup></strong>
  .
  <strong>Menghan Hu<sup>1</sup></strong>
</p>
<p align="center">

<p align="center">
  <strong><sup>1</sup>East China Normal University</strong> &nbsp;&nbsp;&nbsp;
  <strong><sup>2</sup>Shanghai Jiao Tong University</strong> &nbsp;&nbsp;&nbsp;
  <strong><sup>3</sup>Tsinghua Berkeley Shenzhen Institute</strong> &nbsp;&nbsp;&nbsp;
</p>


[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

If you have any questions, please contact Xinxin Zhang(Zhangxinxin5058@163.com) or Menghan Hu(mhhu@ce.ecnu.edu.cn).

## ✨A Gentle Introduction
The QGRM (Quality-Guided Respiration Monitoring) algorithm integrates a two-stage motion artifact suppression module and a quality-guided peak detection module (QGPD). The former module enhances signal stability through FIR
filtering and amplitude limiting, while the latter improves respiration rate estimation by filtering false peaks based on amplitude and zero-crossing constraints. We can achieve the real-time respiration monitoring even in laptop.

This is an overview of the proposed QGRM diagram.
