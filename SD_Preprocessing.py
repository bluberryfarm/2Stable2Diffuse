import os
import shutil
import helper


inputFilePath = "/Users/blumezl1/Documents/SCHOOLYEAR_ASPIRE_2022-2023/Small_Rock_Collection/Real_Rock_TEST/Rocky_Man.jpeg"

outputFile = "/Users/blumezl1/Documents/SCHOOLYEAR_ASPIRE_2022-2023/Small_Rock_Collection/New_Rock_TEST/new_rock_image.jpeg"


def stage1(inputFilePath, outputFilePath, gpuEnabled=False,
           verbose=0):    
    try:
        if not os.path.exists(outputFile):
            os.mkdir(outputFile)
    except OSError:
        print('Directory creation failed')
        success = False
        return success
    if (gpuEnabled):
        NE_return = os.system(
            "python3 denoising/neural-enhance-master/enhance.py "
            "--type=photo --model=repair --zoom=1 --device=cuda " +
            inputFilePath)
        if (NE_return != 0):
            print(
                "\nUnable to run with --device=cuda argument. "
                "Trying --device=cuda*.\n")
            NE_return = os.system(
                "python3 denoising/neural-enhance-master/enhance.py "
                "--type=photo --model=repair --zoom=1 "
                "--device=cuda* " + inputFilePath)
            if (NE_return != 0):
                print(
                    "\nUnable to run with --device=cuda* argument. "
                    "Trying --device=gpu0.\n")
                NE_return = os.system(
                    "python3 "
                    "denoising/neural-enhance-master/enhance.py "
                    "--type=photo --model=repair --zoom=1 "
                    "--device=gpu0 " + inputFilePath)
                if (NE_return != 0):
                    print(
                        "\nUnable to run with --device=gpu0 "
                        "agrument.  Trying to use cpu.\n")
                    NE_return = os.system(
                        "python3 "
                        "denoising/neural-enhance-master/enhance.py "
                        "--type=photo --model=repair --zoom=1 " +
                        inputFilePath)
                    if (NE_return != 0):
                        print("\nUnable to run neural-enhance.\n")
                        success = False
                        return success
    else:
        NE_return = os.system(
            "python3 denoising/neural-enhance-master/enhance.py "
            "--type=photo --model=repair --zoom=1 " + inputFilePath)
        if (NE_return != 0):
            print("\nUnable to run neural-enhance.\n")
            success = False
            return success
    tempFile = inputFilePath.split('.', 1)[0] + "_ne1x" + ".png"
    tempFilename = tempFile.split('/', 1)[-1]
    imageName = inputFilePath.split('/', 1)[-1]
    if (verbose >= 2):
        print("tempFilename = " + tempFilename)
        print("tempFile = " + tempFile)
        print("imageName = " + imageName)
        print("outputFile = " + outputFile)
    if not os.path.exists(tempFile):
        print("De-noising failed!")
        success = False
        return success
    try:
        shutil.move(tempFile, os.path.join(outputFile, tempFilename))
    except helper.PipelineError:
        print("Unable to move file")
        success = False
        return success
    try:
        os.rename(os.path.join(outputFile, tempFilename),
                  outputFilePath)
    except helper.PipelineError:
        print("Unable to rename file")
        success = False
        return success
    return success