import torch
import mytorch
import mytorch.math as math


class LieGroup(mytorch.autograd.Function):
    def __init__(self) -> None:
        super().__init__()
    
    def _create_impl(self, dtype=None, device=None):
        pade_order = 17 if dtype is torch.double else 9
        assert pade_order>=2
        # pade coefficient of (1-e^{-x})/x
        k = torch.arange(1,2*pade_order,dtype=dtype)
        c = -torch.cumprod(-1/k,-1)
        a1,b1 = math.pade(c,pade_order,pade_order)
        # pade coefficient of e^{-x}
        k = torch.arange(2*pade_order-1,dtype=dtype)
        k[0] = 1
        c = -torch.cumprod(-1/k,-1)
        a2,b2 = math.pade(c,pade_order,pade_order)
        
        c = torch.stack(
            [
                torch.stack([a1,b1],dim=-1),
                torch.stack([a2,b2],dim=-1)
            ],
            dim=-1
        ).to(device=device)
        
        class impl(torch.autograd.Function):
            @staticmethod
            def forward(ctx:torch.autograd.function.FunctionCtx, w:torch.Tensor):
                al = self.algebra(w.detach())
                y = torch.matrix_exp(al)
                ctx.save_for_backward(y,w.detach())
                return y
            
            @staticmethod
            def backward(ctx:torch.autograd.function.FunctionCtx, dy:torch.Tensor):
                y,w = ctx.saved_tensors
                ad = self.adjoint(w)
                e = torch.eye(ad.size(-1),dtype=ad.dtype,device=ad.device)
                c_ = c[(...,)+(None,)*ad.ndim]
                # scaling
                n = torch.log2(ad.abs().max()*ad.size(-1)).ceil().int().maximum(torch.tensor(0))
                ad = ad/(2**n)
                # pade approximation
                r = c_[0]*e+c_[1]*ad
                p = ad
                for i in range(2,pade_order):
                    p = ad@p
                    r = r+c_[i]*p
                r = torch.linalg.solve(r[0],r[1])
                # squaring
                for _ in range(n):
                    r[0] = r[0]@(e+r[1])/2
                    r[1] = r[1]@r[1]
                # chain rule
                dw = self.derivative(y, dy.conj(), w)@r[0]
                return self.reg(dw.reshape(w.size()).conj(), w)
            
        return impl.apply
            
    def algebra(self, w:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"LieGroup [{type(self).__name__}] is missing the required \"algebra\" function")
    
    def adjoint(self, w:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"LieGroup [{type(self).__name__}] is missing the required \"adjoint\" function")
    
    def derivative(self, y:torch.Tensor, dy:torch.Tensor, w:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"LieGroup [{type(self).__name__}] is missing the required \"derivative\" function")
    
    def reg(self, dw:torch.Tensor, w:torch.Tensor) -> torch.Tensor:
        return dw


class SO(LieGroup):
    def __init__(self, n:int, device=None) -> None:
        super().__init__()
        self.mdim = n
        self.gdim = n*(n-1)//2
        
        al = torch.zeros(self.gdim,self.mdim,self.mdim,dtype=torch.int)
        self.__index_al = torch.zeros(self.mdim,self.mdim,dtype=torch.long,device=device)
        k = 0
        for i in range(1,n):
            for j in range(i):
                al[k,i,j] = 1
                al[k,j,i] = -1
                self.__index_al[i,j] = k
                self.__index_al[j,i] = k
                k += 1
        
        self.__coef_al = al.sum(0).to(dtype=torch.double,device=device)
        self.__index_al = self.__index_al.flatten()
        
        ad = torch.zeros(self.gdim,self.gdim,self.gdim,dtype=torch.int)
        self.__index_ad = torch.zeros(self.gdim,self.gdim,dtype=torch.long,device=device)
        for i in range(self.gdim):
            ad[i] = self.vectorize(al[i]@al-al@al[i]).transpose(-2,-1)
            self.__index_ad[ad[i]!=0] = i
        
        self.__coef_ad = ad.sum(0).to(dtype=torch.double,device=device)
        self.__index_ad = self.__index_ad.flatten()
        
        self._set(self._create_impl(dtype=torch.double, device=device))
        
    def vectorize(self, x:torch.Tensor):
        idx0,idx1 = mytorch.count_to_index(torch.arange(self.mdim))
        return x[...,idx0,idx1]
            
    def algebra(self, w:torch.Tensor):
        return w[...,self.__index_al].unflatten(-1,(self.mdim,self.mdim))*self.__coef_al
        # return torch.einsum('...i,ijk->...jk',w,self.al)
    
    def adjoint(self, w:torch.Tensor):
        return w[...,self.__index_ad].unflatten(-1,(self.gdim,self.gdim))*self.__coef_ad
        # return torch.einsum('...i,ijk->...jk',w,self.ad)
    
    def derivative(self, y:torch.Tensor, dy:torch.Tensor):
        return torch.zeros(
            y.size()[:-2]+(self.gdim,), 
            dtype=y.dtype, 
            device=y.device
        ).index_add_(
            -1,
            self.__index_al,
            ((y.transpose(-2,-1)@dy)*self.__coef_al).flatten(-2)
        )
        
    