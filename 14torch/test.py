# print("test")
import torch

def test_torch():
    # Check that MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")

    else:
        print("MPS is availabel")
        mps_device = torch.device("mps")
        print(mps_device)

        # Create a Tensor directly on the mps device
        x = torch.ones(5, device=mps_device)
        # Or
        # x = torch.ones(5, device="mps")

        # Any operation happens on the GPU
        y = x * 2
        print(y)

        # Move your model to mps just like any other device
        # model = YourFavoriteNet()
        # model.to(mps_device)

        # Now every call runs on the GPU
        # pred = model(x)



if __name__=="__main__":
    test_torch()