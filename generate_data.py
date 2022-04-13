import bpy
import sys, os, glob
from copy import copy
from mathutils import Matrix
import math
import numpy as np
from numpy import linalg as la
import json
import configargparse
import cv2
import pdb


def fibonacci_sphere(samples=1, R_shift=np.eye(3)):
    """
    Uniformaly (ish) distribute points over a sphere
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    """
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        xyz_rotated = R_shift @ np.reshape([x, y, z], (3,))

        points.append((xyz_rotated[0], xyz_rotated[1], xyz_rotated[2]))

    return points


def get_camera_orientation(camera_location, target_view_point=np.array([0, 0, 0])):
    """
    calculate camera orientation that points camera
    at target_view_point from camera_location
    """
    view_dir = (target_view_point - camera_location) / la.norm(
        target_view_point - camera_location, 2
    )
    z_bar = np.reshape(-view_dir, (3,))
    x_bar = np.reshape([view_dir[1], -view_dir[0], 0], (3,))

    if la.norm(x_bar, 2) < 10 ** (-8):
        x_bar = np.array([1.0, 0.0, 0.0])
    else:
        x_bar /= la.norm(x_bar, 2)

    y_bar = np.cross(z_bar, x_bar)
    R_w_c = Matrix(np.array([x_bar, y_bar, z_bar]).T)

    if np.any(np.isnan(R_w_c)):
        pdb.set_trace()

    eul_w_c_XYZ = R_w_c.to_euler("XYZ")

    return eul_w_c_XYZ, R_w_c


def rotm_and_t_to_tf(R, t):
    tf = np.eye(4)
    tf[0:3, 0:3] = R
    tf[0:3, 3] = np.reshape(t, (3,))
    return tf


def generate_random_R():
    """
    correctly uniformly samples SO(3)
    """
    x = np.random.rand(3)
    v = np.reshape(
        [
            np.cos(2 * math.pi * x[1]) * np.sqrt(x[2]),
            np.sin(2 * math.pi * x[1]) * np.sqrt(x[2]),
            np.sqrt(1 - x[2]),
        ],
        (3, 1),
    )
    H = np.eye(3) - 2 * (v @ v.T)
    ang = 2 * math.pi * x[0]
    Rz = np.array(
        [[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]]
    )
    return -H @ Rz


def gen_data_on_sphere(num_pnts, cam_dist):
    camera_locations_sphere = fibonacci_sphere(num_pnts, generate_random_R())

    for i, cam_loc in enumerate(camera_locations_sphere):
        camera_locations_sphere[i] = (
            np.asarray(camera_locations_sphere[i]) * cam_dist,
            "origin",
            0,
        )

    return camera_locations_sphere


def get_depth_image(b_invert=False, save_fn=None):
    """
    values 0 -> 255, 0 is furthest, 255 is closest
    assumes range map node maps from 0 to 1 values

    set b_invert to True if you want disparity image
    """
    raw = np.asarray(bpy.data.images["Viewer Node"].pixels)
    scene = bpy.data.scenes["Scene"]
    raw = np.reshape(raw, (scene.render.resolution_x, scene.render.resolution_y, 4))
    raw = raw[:, :, 0]
    raw = np.flipud(raw)
    raw1 = 1  # bpy.data["Map Range Node"].from_min
    raw0 = 0
    depth0 = bpy.data.objects["Camera"].data.clip_start
    depth1 = bpy.data.objects["Camera"].data.clip_end

    # back_mask = raw == raw1
    depth = ((raw - raw0) / (raw1 - raw0)) * (depth1 - depth0) + depth0
    if b_invert:
        depth = 1.0 / depth

    # depth_img = (1.0 - depth / depth1*255).astype(np.uint8)

    if not save_fn is None:
        if save_fn[-3:] == "npy":
            np.save(save_fn, depth)
        else:
            # depth_cv2 = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2BGR)
            # cv2.imwrite(save_fn, depth_cv2)
            cv2.imwrite(save_fn, depth)
    return depth


def nerf_object_training_data_generator(object_name, save_dir, prms):
    # Parameter
    obj = bpy.data.objects[object_name]
    print("\n--------------------   STARTING BLENDER SCRIPT   --------------------\n")
    camera = bpy.data.objects["Camera"]
    for mode, num_imgs in zip(
        ["train", "test", "val"], [prms["num_train"], prms["num_test"], prms["num_val"]]
    ):

        print("\n-------------------- Starting {} --------------------".format(mode))
        # Create struct for data to be written to json
        json_data = {}

        if camera.data.clip_end > 10:
            for i in range(10):
                print(
                    "WARNING... set camera clip start & end to desired"
                    + "near/far bounds or else edit in json file after!!!"
                )

        json_data["far"] = camera.data.clip_end
        json_data["near"] = np.max([camera.data.clip_start, 0])
        json_data["obj_max_dim"] = la.norm(
            np.array(obj.dimensions)
        )  # longest dist through object
        json_data["camera_angle_x"] = camera.data.angle_x  # in radians
        json_data["frames"] = []

        # concatenate paths, ensure directories exist, and remove existing files
        output_dir_images = os.path.abspath(save_dir + "/nerf_training_data/" + mode)
        os.makedirs(output_dir_images, exist_ok=True)
        existing_files = glob.glob(output_dir_images + "/*.png")
        for f in existing_files:
            os.remove(f)
        output_dir_json = os.path.abspath(
            save_dir + "/nerf_training_data/transforms_" + mode + ".json"
        )
        output_file_pattern_string = "render{:03d}.png"

        # First choose x/y/z location of camera origins
        camera_locations = []
        if mode == "train":
            for i in [-1, 1]:
                for j in [-1, 1]:
                    for k in [-1, 1]:
                        xyz_corner = np.array(obj.dimensions) / 2.0 * [i, j, k]
                        xyz_dir = xyz_corner / la.norm(xyz_corner)
                        camera_locations.append(
                            (xyz_dir * prms["cam_offset_far"], "origin", 0)
                        )

            x_span, y_span, z_span = np.array(obj.dimensions)
            dim_arr = [(y_span, z_span), (x_span, z_span), (x_span, y_span)]
            dim2_arr = [x_span, y_span, z_span]

            for i, (dim0, dim1) in enumerate(dim_arr):
                planar_coords = np.array([dim0, dim1]) / 2.0
                dim2 = obj.dimensions[i] / 2.0 + prms["cam_offset_far"]
                camera_locations.extend(
                    [
                        (xyz, "origin", 0)
                        for xyz in np.insert(
                            np.zeros((2, 2)), i, dim2 * np.array([1, -1]), axis=0
                        ).T
                    ]
                )

            camera_locations.extend(
                gen_data_on_sphere(
                    num_pnts=(num_imgs - len(camera_locations)),
                    cam_dist=prms["cam_offset_far"],
                )
            )

        else:  #  (or non-training run for rect)
            # calculate even spread of locations around unit sphere
            camera_locations = gen_data_on_sphere(
                num_pnts=num_imgs, cam_dist=prms["cam_offset_far"]
            )

        # Next, actually render the images at each location
        org = np.array([0, 0, 0])
        for step, (camera_loc, orient_mode, axis_to_zero) in enumerate(
            camera_locations
        ):
            frame = {}
            print("\n-----{} image {} / {} -----".format(mode, step, num_imgs - 1))

            # Calc camera direction
            if orient_mode == "origin":
                # calculate camera orientation that points to origin from camera location
                _, R_w_c = get_camera_orientation(camera_loc, org)
            else:
                # make - z axis point towards mesh
                org = copy(camera_loc)
                org[axis_to_zero] = 0
                _, R_w_c = get_camera_orientation(camera_loc, org)
            T_w_c = rotm_and_t_to_tf(R_w_c, camera_loc)

            frame["transform_matrix"] = T_w_c.tolist()
            frame["file_path"] = os.path.join(
                "./", mode, output_file_pattern_string[:-4].format(step)
            )

            # update camera location
            print("cam. T_w_c. = \n{}".format(T_w_c))
            camera.matrix_world = Matrix(T_w_c)

            # save image from camera
            bpy.context.scene.render.filepath = os.path.join(
                output_dir_images, output_file_pattern_string.format(step)
            )
            bpy.ops.render.render(write_still=True)
            json_data["frames"].append(frame)

            b_save_depth = True
            if b_save_depth:
                depth_path = os.path.join(output_dir_images, "depth")
                os.makedirs(depth_path, exist_ok=True)
                get_depth_image(
                    save_fn=os.path.join(
                        depth_path, "depth_render{:03d}.npy".format(step)
                    )
                )
                # get_depth_image(save_fn=os.path.join(
                #    depth_path, "depth_render{:d}.png".format(step)))

        # write json file
        print("Done collecting image, now writing json file")
        with open(output_dir_json, "w", encoding="utf-8") as outfile:
            json.dump(json_data, outfile)


if __name__ == "__main__":
    np.set_printoptions(
        linewidth=160, suppress=True
    )  # format numpy so printing matrices is more clear
    print(
        "Starting nerf_object_training_data_generator [running python {}]".format(
            sys.version_info[0]
        )
    )
    # bpy.context.scene.render.image_settings.file_format = "PNG" # set to 'PNG' or 'JPG' to control file output format

    # TO DO: Automate choice of camera locations. There should be a close offset and a sphere offset. The close offset
    #        should take zoomed in pictures normal to the mesh the cover the full surface. The sphere offset should be
    #        what I have now, where the camera moves around and sees the whole object.
    # sample_surface = rect is currently a work around for this for the special case of box-shaped objects (except my
    #   camera orientations still point to the origin - it should prob be normal to the object)

    # if choosing rect for sample_surface, this is only used for training data generation. Give 2 distances, one for training
    #   data offset from the surface and the other for testing images which will use the sphere surface
    # num_train should be 14 + extra - 6 normal to each face of the bounding box and 8 at each corner of bounding box + extra scattered randomly

    # cam_offset_near - used for bounding box surface offset to see close up detail this, + num_train should see all of the outside of the image
    # cam_offset_far - used for sphere radius from origin for images seeing full object (choose so most, if not all, of object is in FOV)

    prms = {
        "teddy_bear": {
            "cam_offset_near": 0.55,
            "cam_offset_far": 0.75,
            "num_train": 14 + 100,
            "num_test": 10,
            "num_val": 2,
        },
        "dining_table": {
            "cam_offset_near": 0.65,
            "cam_offset_far": 3.65,
            "num_train": 14 + 175,
            "num_test": 10,
            "num_val": 2,
        },
        "laptop": {
            "cam_offset_near": 0.50,
            "cam_offset_far": 0.75,
            "num_train": 14 + 100,
            "num_test": 10,
            "num_val": 2,
        },
        "mug": {
            "cam_offset_near": 0.30,
            "cam_offset_far": 0.30,
            "num_train": 14 + 100,
            "num_test": 10,
            "num_val": 2,
        },
    }
    # "mug"         : {"cam_offset_near" : 2.95, "cam_offset_far" : 2.95, "num_train" : 14+100, "num_test": 10, "num_val" : 2}}
    # "stonehenge"  : {"cam_dist" : [0.1, 0.3, 0.7, 1.1, 1.5], "num_train" : 500, "num_test": 20, "num_val" : 10}}

    object_name = None
    for obj_name in prms.keys():
        if obj_name in bpy.data.objects.keys():
            object_name = obj_name
            break

    if object_name is None or np.any(
        np.abs(bpy.data.objects[object_name].dimensions) < 10 ** -7
    ):
        print(
            "No recognizable objects in the scene, make sure info is added to the dictionary!\n"
        )
    else:
        print("Generating data for object: {}".format(object_name))
        data_dir = "/home/adam/Documents/song/single_object_blenders/"
        nerf_object_training_data_generator(
            prms=prms[object_name],
            object_name=object_name,
            save_dir=os.path.join(os.path.dirname(data_dir), object_name),
        )
    print(
        "--------------------    DONE NERF TRAINING DATA SCRIPT    --------------------"
    )
