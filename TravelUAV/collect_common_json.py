import os
import shutil


episodes = [
    "0a7a07a1-0757-4528-9463-ca6c0eb2ec92",
    "20375e27-9b65-45a3-bfd8-4a79b1069cf5",
    "4da06af0-95b7-4fd3-8c21-c31315857138",
    "aa137a79-c5b8-41b3-b4f8-2532767cdd85",
    "ac96c2d1-4fa6-4b4b-b41a-0e4b97f07303"
]


save_root="/mnt/sda2/TravelUAV/common_success_5tasks"



sources={

"TravelUAV":[
"/mnt/sda2/TravelUAV/result(100-300)/result(v=1)",
"/mnt/sda2/TravelUAV/result（100米以内29个）/result（v=1）"
],


"TravelUAV_TTS":[
"/mnt/sda2/TravelUAV/result_tts_ing"
]

}



def find_task(base,eid):

    root=os.path.join(
        base,
        "eval_closeloop",
        "eval_test"
    )

    for name in [
        "success_"+eid,
        eid
    ]:

        path=os.path.join(root,name)

        if os.path.exists(path):
            return path

    return None



def copy_file(src,dst):

    os.makedirs(
        os.path.dirname(dst),
        exist_ok=True
    )

    shutil.copy2(
        src,
        dst
    )



for method,bases in sources.items():

    print("\n==========")
    print(method)


    for eid in episodes:


        save_dir=os.path.join(
            save_root,
            method,
            eid
        )

        found=False



        for base in bases:


            task=find_task(base,eid)


            if task is None:
                continue


            print(
                "Found:",
                method,
                eid,
                task
            )


            found=True



            # =====================
            # evaluation.json
            # =====================

            eva=os.path.join(
                task,
                "evaluation.json"
            )

            if os.path.exists(eva):

                copy_file(
                    eva,
                    os.path.join(
                        save_dir,
                        "evaluation.json"
                    )
                )



            # =====================
            # log
            # =====================

            log=os.path.join(
                task,
                "log"
            )


            if os.path.exists(log):

                shutil.copytree(
                    log,
                    os.path.join(
                        save_dir,
                        "log"
                    ),
                    dirs_exist_ok=True
                )



            # =====================
            # TravelUAV timing
            # =====================

            timing=os.path.join(
                base,
                "timing",
                eid
            )


            if os.path.exists(timing):

                shutil.copytree(
                    timing,
                    os.path.join(
                        save_dir,
                        "timing"
                    ),
                    dirs_exist_ok=True
                )



            # =====================
            # TTS timing
            # =====================

            tts_timing=os.path.join(
                base,
                "eval_closeloop",
                "eval_test",
                "timing",
                eid,
                "timing.json"
            )


            if os.path.exists(tts_timing):

                copy_file(
                    tts_timing,
                    os.path.join(
                        save_dir,
                        "timing.json"
                    )
                )


            break



        if not found:

            print(
                "NOT FOUND:",
                method,
                eid
            )



print("\nFinished.")