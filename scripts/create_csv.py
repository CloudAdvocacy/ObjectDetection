import mPyPl as mp
import os

data_dir = "/mnt/data/HollywoodHeads"

def fint(x):
    return int(float(x))

@mp.Pipe
def import_fields(seq,field_name):
    for x in seq:
        try:
            for k in x[field_name].keys():
                x[k] = x[field_name][k]
            yield x
        except:
            print("WARN: object field is {}".format(x[field_name]))
            pass
(
    mp.get_pascal_annotations(os.path.join(data_dir,"Annotations")) 
  #| mp.take(100)
  | mp.unroll("object")
  | mp.apply("filename","fname",lambda x: os.path.join(data_dir,"JPEGImages",x))
  | import_fields('object')
  | mp.apply('bndbox_xmin','xmin',fint)
  | mp.apply('bndbox_xmax','xmax',fint)
  | mp.apply('bndbox_ymin','ymin',fint)
  | mp.apply('bndbox_ymax','ymax',fint)
  | mp.select_fields(['fname','xmin','ymin','xmax','ymax','name'])
  | mp.write_csv('annotations.csv',write_headers=False))
