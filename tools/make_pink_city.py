"""Build the City of Pink USD (3 x 3 m town for Pinky Pro) from the textures in assets/pink_city/textures/.

  python tools/make_pink_city.py      # -> assets/pink_city/pink_city.usd, pink_city_floor.usd

Same layout as 16_pink_city/pink_city.py of https://github.com/PinkWink/nvidia-dli-lecture (the textures are made
there): ring road + cross road painted on the floor, walls, buildings and trees with colliders.
  pink_city.usd        everything (play, dataset)
  pink_city_floor.usd  only the painted floor (training: thousands of copies, nothing to collide with)
Written with plain USD (UsdPreviewSurface, relative texture paths) so the folder can be moved.
"""
import math

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from rl_shortcut import ROOT, shutdown  # noqa: E402

OUT = ROOT / "assets/pink_city"
SIZE, ROAD, RING = 3.0, 0.40, 1.05
HALF = SIZE / 2
BLOCK_MIN, BLOCK_MAX = ROAD / 2, RING - ROAD / 2
WALL_HEIGHT, WALL_THICKNESS = 0.35, 0.04
PINK, DEEP_PINK, LIGHT_PINK = (236, 112, 170), (214, 72, 140), (250, 214, 228)
CITY = "/PinkCity"


def linear(color):
    return Gf.Vec3f(*[(c / 255.0) ** 2.2 for c in color])


class Builder:
    def __init__(self, path):
        self.stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.stage.SetDefaultPrim(UsdGeom.Xform.Define(self.stage, CITY).GetPrim())
        self.looks = {}

    def look(self, name, color=None, texture=None, glow=False, roughness=0.7):
        if name in self.looks:
            return self.looks[name]
        path = f"{CITY}/Looks/{name}"
        material = UsdShade.Material.Define(self.stage, path)
        shader = UsdShade.Shader.Define(self.stage, path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        if texture:
            reader = UsdShade.Shader.Define(self.stage, path + "/st")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            image = UsdShade.Shader.Define(self.stage, path + "/Texture")
            image.CreateIdAttr("UsdUVTexture")
            image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./textures/{texture}")
            image.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            image.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                reader.ConnectableAPI(), "result")
            rgb = image.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
            if glow:        # signs stay readable in the shade
                shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
        else:
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(linear(color))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        self.looks[name] = material
        return material

    def mesh(self, path, points, face_counts, face_indices, uvs, look, solid):
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
        mesh.CreateFaceVertexCountsAttr(face_counts)
        mesh.CreateFaceVertexIndicesAttr(face_indices)
        mesh.CreateSubdivisionSchemeAttr("none")
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                     UsdGeom.Tokens.faceVarying)
        st.Set([Gf.Vec2f(*uv) for uv in uvs])
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(look)
        if solid:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        return mesh

    def box(self, path, center, size, look, solid=True):
        """Box with the texture once on each side face."""
        (x, y, z), (sx, sy, sz) = center, size
        c = [(x + dx * sx / 2, y + dy * sy / 2, z + dz * sz / 2)
             for dz in (-1, 1) for dy in (-1, 1) for dx in (-1, 1)]       # index = dx + 2 dy + 4 dz
        faces = [(0, 1, 5, 4), (1, 3, 7, 5), (3, 2, 6, 7), (2, 0, 4, 6), (4, 5, 7, 6), (0, 2, 3, 1)]
        quad = [(0, 0), (1, 0), (1, 1), (0, 1)]
        return self.mesh(path, c, [4] * 6, [i for f in faces for i in f], quad * 6, look, solid)

    def board(self, path, center, width, height, look, facing):
        """Upright picture seen from the side `facing` [deg] (0 = from +X, 90 = from +Y)."""
        f = math.radians(facing)
        r = (-math.sin(f), math.cos(f))                      # the viewer's right
        x, y, z = center
        pts = [(x + s * r[0] * width / 2, y + s * r[1] * width / 2, z + t * height / 2)
               for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        return self.mesh(path, pts, [4], [0, 1, 2, 3], [(0, 0), (1, 0), (1, 1), (0, 1)], look, False)

    def flat(self, path, center, size, look):
        """Picture lying on the ground or a roof: image top = +Y."""
        (x, y, z), (sx, sy) = center, size
        pts = [(x + s * sx / 2, y + t * sy / 2, z) for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        return self.mesh(path, pts, [4], [0, 1, 2, 3], [(0, 0), (1, 0), (1, 1), (0, 1)], look, False)

    def round(self, kind, path, center, radius, height, look):
        shape = getattr(UsdGeom, kind).Define(self.stage, path)
        shape.CreateAxisAttr("Z")
        shape.CreateRadiusAttr(radius)
        shape.CreateHeightAttr(height)
        shape.AddTranslateOp().Set(Gf.Vec3d(*center))
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(look)
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())

    # ------------------------------------------------------------------ the city
    def floor(self):
        self.flat(f"{CITY}/Floor", (0, 0, 0.001), (SIZE, SIZE), self.look("Floor", texture="floor.png", roughness=0.8))

    def walls(self):
        wall, cap = self.look("Wall", color=(246, 240, 243)), self.look("WallCap", color=PINK)
        city = self.look("SignCity", texture="sign_city.png", glow=True)
        lab = self.look("SignPinklab", texture="sign_pinklab.png", glow=True)
        e, long = HALF + WALL_THICKNESS / 2, SIZE + 2 * WALL_THICKNESS
        for name, (x, y), (sx, sy), facing, sign in (("North", (0, e), (long, WALL_THICKNESS), -90.0, city),
                                                     ("South", (0, -e), (long, WALL_THICKNESS), 90.0, lab),
                                                     ("East", (e, 0), (WALL_THICKNESS, long), 180.0, lab),
                                                     ("West", (-e, 0), (WALL_THICKNESS, long), 0.0, city)):
            self.box(f"{CITY}/Walls/{name}", (x, y, WALL_HEIGHT / 2), (sx, sy, WALL_HEIGHT), wall)
            self.box(f"{CITY}/Walls/{name}Cap", (x, y, WALL_HEIGHT + 0.01), (sx + 0.01, sy + 0.01, 0.02), cap, solid=False)
            d = WALL_THICKNESS / 2 + 0.002
            self.board(f"{CITY}/Signs/{name}", (x + d * math.cos(math.radians(facing)), y + d * math.sin(math.radians(facing)), 0.19),
                       1.0, 0.25, sign, facing)

    def building(self, name, center, size, look, roof):
        (x, y), (sx, sy, h) = center, size
        self.box(f"{CITY}/Buildings/{name}", (x, y, h / 2), (sx, sy, h), look)
        self.box(f"{CITY}/Buildings/{name}Roof", (x, y, h + 0.008), (sx + 0.02, sy + 0.02, 0.016), roof, solid=False)

    def roof_sign(self, name, center, roof, width, height, look, along_x):
        (x, y), z = center, roof + 0.02 + height / 2
        frame = self.look("BoardFrame", color=(70, 72, 84))
        size = (width + 0.02, 0.012, height + 0.02) if along_x else (0.012, width + 0.02, height + 0.02)
        self.box(f"{CITY}/Buildings/{name}SignFrame", (x, y, z), size, frame, solid=False)
        for k, facing in enumerate((-90.0, 90.0) if along_x else (180.0, 0.0)):
            f = math.radians(facing)
            self.board(f"{CITY}/Signs/{name}Roof{k}", (x + 0.008 * math.cos(f), y + 0.008 * math.sin(f), z), width, height, look, facing)

    def tree(self, name, x, y, height):
        self.round("Cylinder", f"{CITY}/Park/{name}Trunk", (x, y, height * 0.25), 0.015, height * 0.5,
                   self.look("Trunk", color=(128, 92, 70)))
        self.round("Cone", f"{CITY}/Park/{name}Leaf", (x, y, height * 0.7), 0.065, height * 0.7,
                   self.look("Leaf", color=(74, 150, 96)))

    def city(self):
        self.floor()
        self.walls()
        pink = self.look("WindowsPink", texture="windows_pink.png")
        white = self.look("WindowsWhite", texture="windows_white.png")
        blue = self.look("WindowsBlue", texture="windows_blue.png")
        roof_dark, roof_pink = self.look("RoofDark", color=(84, 86, 98)), self.look("RoofPink", color=DEEP_PINK)
        mid = (BLOCK_MIN + BLOCK_MAX) / 2
        # upper right: PinkLAB headquarters
        self.building("Headquarters", (mid, mid), (0.42, 0.42, 0.55), pink, roof_dark)
        self.roof_sign("Headquarters", (mid, mid), 0.566, 0.39, 0.13,
                       self.look("SignHeadquarters", texture="sign_headquarters.png", glow=True), along_x=True)
        self.flat(f"{CITY}/Signs/HeadquartersRoofLogo", (mid, mid + 0.09, 0.568), (0.2, 0.2),
                  self.look("LogoBoard", texture="logo_board.png"))
        # upper left: Pink Academy and a shop
        academy = self.look("SignAcademy", texture="sign_academy.png", glow=True)
        self.building("PinkAcademy", (-mid - 0.13, mid), (0.22, 0.46, 0.26), white, roof_pink)
        self.roof_sign("PinkAcademy", (-mid - 0.13, mid), 0.276, 0.42, 0.095, academy, along_x=False)
        self.board(f"{CITY}/Signs/PinkAcademyDoor", (-mid - 0.13, mid - 0.232, 0.045), 0.20, 0.045, academy, -90.0)
        self.building("Shop", (-mid + 0.15, mid + 0.08), (0.24, 0.30, 0.38), blue, roof_dark)
        # lower right: apartments
        self.building("ApartmentA", (mid, -mid - 0.14), (0.48, 0.20, 0.44), white, roof_pink)
        self.building("ApartmentB", (mid + 0.14, -mid + 0.10), (0.20, 0.26, 0.32), pink, roof_dark)
        # lower left: park
        for k, (x, y) in enumerate(((-0.74, -0.30), (-0.52, -0.30), (-0.30, -0.33), (-0.74, -0.74), (-0.30, -0.62))):
            self.tree(f"Tree{k}", x, y, 0.20 + 0.03 * (k % 3))
        self.building("Kiosk", (-0.33, -0.76), (0.14, 0.12, 0.13), self.look("Kiosk", color=LIGHT_PINK), roof_pink)

    def save(self):
        self.stage.GetRootLayer().Save()
        print("[city]", self.stage.GetRootLayer().realPath)


full = Builder(OUT / "pink_city.usd")
full.city()
full.save()
floor = Builder(OUT / "pink_city_floor.usd")
floor.floor()
floor.save()
shutdown(app)
