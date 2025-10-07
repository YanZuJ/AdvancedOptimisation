1. Download Matlab R2024a
2. Download CVXPY and on Home > Get Path, set your Path to the CVXX folder (C:\Users\(User)\cvx)
3. Get Academic License https://www.mosek.com/products/academic-licenses/ and get the license via your school email. On your school email, download the .lic file and then replace the .lic file in Mosek folder.
4. Set the Path of the Mosel path to: C:\Users\(User)\Mosek\10.1\tools\platform\win64x86\bin . Depending on the User profile on your computer, replace accordingly
5. On an live script Matlab file, do cvx_setup

Files to note:
1. Q75231.m

Input: Original Image
Output: 2 images (CVX and ADMM) and some graph

2. NoisedInput.m

Image: Noised Input
Output: Greyed ADMM Output
