Nested namedtuples that hold the data for the paper

Data
    Synt
        Intr         # listo: SyntIntr
            camera  'vcaWide' string camera model
            model   string indicating camera intrinsic model
                    ['poly', 'rational', 'fisheye', 'stereographic']
            s       is the image size
            k       sintehtic stereographic parameter
            uv      = s / 2 is the stereographic optical center
        Ches         # listo: SyntChes
            nIm     number of images
            nPt     number of point in image
            objPt   chessboard model grid
            rVecs   synth rotation vectors
            tVecs   synth tVecs
            imgPt   synth corners projected from objPt with synth params
            imgNse  noise of 1 sigma for the image
        Extr         # listo: SyntExtr
            ang     angles of synth pose tables
            h       heights  of synth pose tables
            rVecs   rotation vectors associated to angles
            tVecs   tVecs associated to angles and h
            objPt   distributed 3D points on the floor
            imgPt   projected to image
            imgNse  noise for image detection, sigma 1
            index10 indexes to select 10 points well distributed
    Real
        Ches         # listo: RealChes
            nIm     number of chess images
            nPt     number of chess points per image
            objPt   chessboard model, 3D
            imgPt   detected corners in chessboard images
            imgFls  list of paths to the chessboard images
        Balk
            objPt   calibration world points, lat lon
            imgPt   image points for calibration
            priLLA  prior lat-lon-altura
            imgFl   camera snapshot file
        Dete
            carGps  car gps coordinates
            carIm   car image detection traces

