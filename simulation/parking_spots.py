"""Numbered open bays in the authored parking lot (not sensor clearance)."""
import math

# Matches the parked vehicles in build_scene.py. IDs run left to right along
# the near row first, then the building row, in world X coordinates.
PARKED_X={-1:(-14.,0.,8.4,14.),1:(-16.8,-11.2,-5.6,2.8,8.4,14.)}
SPOTS=[]
for row in (-1,1):
    for index in range(12):
        x=round(-16.8+index*2.8,1)
        if any(abs(x-t)<.01 for t in PARKED_X[row]):continue
        SPOTS.append({'id':len(SPOTS)+1,'x':x,'y':row*6.9,
                      'heading':row*math.pi/2,'row':'near' if row<0 else 'building'})
DEFAULT_SPOT=8  # Near row, between the red and white cars, as marked by the user.


def get_spot(number):
    return next(spot for spot in SPOTS if spot['id']==number)


def parking_heading(spot,mode):
    angle=spot['heading']+(math.pi if mode=='back-in' else 0.)
    return (angle+math.pi)%(2*math.pi)-math.pi


def draw_spots(stage,selected):
    """Floor numbers and selected bay outline, hidden with other driving debug."""
    from pxr import Gf,UsdGeom
    root='/World/DrivingDebug/ParkingSpots'
    stage.RemovePrim(root)
    segments={
        'a':((0,1),(1,1)),'b':((1,1),(1,.5)),
        'c':((1,.5),(1,0)),'d':((0,0),(1,0)),
        'e':((0,0),(0,.5)),'f':((0,.5),(0,1)),
        'g':((0,.5),(1,.5))}
    digits={'0':'abcdef','1':'bc','2':'abged','3':'abgcd','4':'fgbc',
            '5':'afgcd','6':'afgecd','7':'abc','8':'abcdefg','9':'abfgcd'}
    for spot in SPOTS:
        color=(.25,1.,.4) if spot['id']==selected else (1.,.9,.35)
        number=str(spot['id']);points=[]
        for i,digit in enumerate(number):
            for segment in digits[digit]:
                for x,y in segments[segment]:
                    points.append(Gf.Vec3f(spot['x']+(i-(len(number)-1)/2)*.55+(x-.5)*.38,
                                          math.copysign(4.25,spot['y'])+(y-.5)*.75,.085))
        curve=UsdGeom.BasisCurves.Define(stage,f'{root}/Spot{spot["id"]}')
        curve.CreateTypeAttr('linear');curve.CreateCurveVertexCountsAttr([2]*(len(points)//2))
        curve.CreatePointsAttr(points);curve.CreateWidthsAttr([.065])
        curve.SetWidthsInterpolation('constant');curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        if spot['id']==selected:
            x,y=spot['x'],spot['y']
            outline=UsdGeom.BasisCurves.Define(stage,root+'/SelectedBay')
            outline.CreateTypeAttr('linear');outline.CreateCurveVertexCountsAttr([5])
            outline.CreatePointsAttr([Gf.Vec3f(x+dx,y+dy,.085) for dx,dy in
                                     [(-1.3,-2.35),(1.3,-2.35),(1.3,2.35),(-1.3,2.35),(-1.3,-2.35)]])
            outline.CreateWidthsAttr([.07]);outline.SetWidthsInterpolation('constant')
            outline.CreateDisplayColorAttr([Gf.Vec3f(*color)])
