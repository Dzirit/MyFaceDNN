using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Intel.RealSense;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using Accord.MachineLearning;

namespace MyFaceDNN
{
    public class FaceProcces
    {
        
        private Net faceDetector,embedder, embedder2 = null;
        private FacemarkLBF facemark = null;
        private Mat dbFaceVector;
        //private EigenFaceRecognizer recognizer;
        //private VectorOfMat imageList = new VectorOfMat();
        //private List<string> nameList = new List<string>();
        //private VectorOfInt labelList = new VectorOfInt();
        public FaceProcces()
        {
            if (facemark == null)
            {
                using (FacemarkLBFParams facemarkParam = new FacemarkLBFParams())
                {
                    facemark = new FacemarkLBF(facemarkParam);
                    facemark.LoadModel("lbfmodel.yaml");
                }
            }
           
            faceDetector = DnnInvoke.ReadNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel");
            embedder = DnnInvoke.ReadNet("nn4.small2.v1.t7");
            embedder2 = DnnInvoke.ReadNet("nn4.small2.v1.t7");
            var dbFace = GetPhoto();
            dbFaceVector = GetFeatureVector2(dbFace);
            /////
            /// int i = 0;
            //labelList.Push(new[] { i++ });
            //imageList.Push(dbFaceVector);
            //recognizer = new EigenFaceRecognizer(imageList.Size);
            //recognizer.Train(imageList, labelList);

            //Accord.MachineLearning.VectorMachines.MultilabelSupportVectorMachine
        }



        public void DetectAndRender(Mat image, DepthFrame depthFrame = null)
        {

            var fullFaceRegions= DetectFace(image);
            //if (partialFaceRegions.Count > 0)
            //{
            //    foreach (Rectangle face in partialFaceRegions)
            //    {
            //        CvInvoke.Rectangle(image, face, new MCvScalar(0, 0, 255));
            //    }
            //}

            if (fullFaceRegions.Count > 0)
            {
                foreach (Rectangle face in fullFaceRegions)
                {
                    var faceVector = GetFeatureVector(new Mat(image, face));
                    VerifyFace(faceVector, dbFaceVector);
                    CvInvoke.Rectangle(image, face, new MCvScalar(0, 255, 0));
                }

                using (VectorOfRect vr = new VectorOfRect(fullFaceRegions.ToArray()))
                using (VectorOfVectorOfPointF landmarks = new VectorOfVectorOfPointF())
                {
                    facemark.Fit(image, vr, landmarks);

                    for (int i = 0; i < landmarks.Size; i++)
                    {
                        using (VectorOfPointF vpf = landmarks[i])
                        {
                            var good = new MCvScalar(0, 255, 0);
                            var bad = new MCvScalar(0, 0, 255);
                            var color = new MCvScalar(255, 0, 0);
                            if (depthFrame != null)
                            {
                                color = ValidateFace(depthFrame, vpf) ? good : bad;
                            }
                            FaceInvoke.DrawFacemarks(image, vpf, color);
                        }

                    }
                }
            }
        }

        private Mat GetFeatureVector(Mat face)
        {
            int imgDim = 96;
            MCvScalar meanVal = new MCvScalar(0, 0, 0);
            Size imageSize = face.Size;
            Mat inputBlob = DnnInvoke.BlobFromImage(
                face,
                1.0/255,
                new Size(imgDim, imgDim),
                meanVal,
                true,
                false);

            embedder.SetInput(inputBlob);
            inputBlob.Dispose();
            return embedder.Forward();
        }
        private Mat GetFeatureVector2(Mat face)
        {
            int imgDim = 96;
            MCvScalar meanVal = new MCvScalar(0, 0, 0);
            Size imageSize = face.Size;
            Mat inputBlob = DnnInvoke.BlobFromImage(
                face,
                1.0 / 255,
                new Size(imgDim, imgDim),
                meanVal,
                true,
                false);

            embedder2.SetInput(inputBlob);
            inputBlob.Dispose();
            return embedder2.Forward();
        }

        private List<Rectangle> DetectFace(Mat image)
        {
            int imgDim = 300;
            MCvScalar meanVal = new MCvScalar(104, 177, 123);

            Size imageSize = image.Size;
            using (Mat inputBlob = DnnInvoke.BlobFromImage(
                image,
                1.0,
                new Size(imgDim, imgDim),
                meanVal,
                false,
                false))
                faceDetector.SetInput(inputBlob, "data");

            using (Mat detection = faceDetector.Forward("detection_out"))
            {
                float confidenceThreshold = 0.5f;

                List<Rectangle> fullFaceRegions = new List<Rectangle>();
                List<Rectangle> partialFaceRegions = new List<Rectangle>();
                Rectangle imageRegion = new Rectangle(Point.Empty, image.Size);

                float[,,,] values = detection.GetData(true) as float[,,,];
                for (int i = 0; i < values.GetLength(2); i++)
                {
                    float confident = values[0, 0, i, 2];

                    if (confident > confidenceThreshold)
                    {
                        float xLeftBottom = values[0, 0, i, 3] * imageSize.Width;
                        float yLeftBottom = values[0, 0, i, 4] * imageSize.Height;
                        float xRightTop = values[0, 0, i, 5] * imageSize.Width;
                        float yRightTop = values[0, 0, i, 6] * imageSize.Height;
                        RectangleF objectRegion = new RectangleF(
                            xLeftBottom,
                            yLeftBottom,
                            xRightTop - xLeftBottom,
                            yRightTop - yLeftBottom);
                        Rectangle faceRegion = Rectangle.Round(objectRegion);

                        if (imageRegion.Contains(faceRegion))
                            fullFaceRegions.Add(faceRegion);
                        else
                        {
                            partialFaceRegions.Add(faceRegion);
                        }
                    }
                }
                return fullFaceRegions;
            }
        }

        private void VerifyFace(Mat face1, Mat face2)
        {
            ///////////////////
            //var d=recognizer.Predict(face1);
            //Console.SetCursorPosition(0, 7);
            //Console.WriteLine($"Distance:{d.Distance} label:{d.Label}");
            ////////
            //var distance = CvInvoke.Norm(face1, face2, Emgu.CV.CvEnum.NormType.L2);
            //Console.SetCursorPosition(0, 8);
            //Console.WriteLine($"Distance:{Math.Abs(distance)*100}");
        }

        private Mat GetPhoto()
        {
            Mat image = new Mat(".\\1.jpg");
            var fullFaceRegions = DetectFace(image);
            if (fullFaceRegions.Count>0)
            {
                return new Mat(image, fullFaceRegions[0]);
            }
            else
            {
                throw new Exception("No face detected in db");
            }
        }

        #region face_markup
        /*
    The 68-point annotations for the iBUG 300-W face landmark dataset.
    See this picture:
    https://ibug.doc.ic.ac.uk/media/uploads/images/annotpics/figure_68_markup.jpg
    NOTE: the indexes in the picture are 1-based, so the actual C++ indexes are less 1.
    NOTE: "Right" and "left" refer to the face being described, so are the mirror of the
    side that an onlooker (from the front) would see.
    */
        public enum markup_68
        {
            // Starting with right ear, the jaw [1-17]
            RIGHT_EAR,
            JAW_FROM = RIGHT_EAR,
            RIGHT_JAW_FROM = RIGHT_EAR,
            RIGHT_1,
            RIGHT_2,
            RIGHT_3,
            RIGHT_4,
            RIGHT_5,
            RIGHT_6,
            RIGHT_7,
            RIGHT_JAW_TO = RIGHT_7,
            CHIN,
            CHIN_FROM = CHIN - 1,
            CHIN_TO = CHIN + 1,
            LEFT_7 = CHIN + 1,
            LEFT_JAW_FROM = LEFT_7,
            LEFT_6,
            LEFT_5,
            LEFT_4,
            LEFT_3,
            LEFT_2,
            LEFT_1,
            LEFT_EAR,
            LEFT_JAW_TO = LEFT_EAR,
            JAW_TO = LEFT_EAR,

            // Eyebrows [18-22] and [23-27]
            RIGHT_EYEBROW_R,
            RIGHT_EYEBROW_FROM = RIGHT_EYEBROW_R,
            RIGHT_EYEBROW_1,
            RIGHT_EYEBROW_2,
            RIGHT_EYEBROW_3,
            RIGHT_EYEBROW_L,
            RIGHT_EYEBROW_TO = RIGHT_EYEBROW_L,
            LEFT_EYEBROW_R,
            LEFT_EYEBROW_FROM = LEFT_EYEBROW_R,
            LEFT_EYEBROW_1,
            LEFT_EYEBROW_2,
            LEFT_EYEBROW_3,
            LEFT_EYEBROW_L,
            LEFT_EYEBROW_TO = LEFT_EYEBROW_L,

            // Nose [28-36]
            NOSE_RIDGE_TOP,
            NOSE_RIDGE_FROM = NOSE_RIDGE_TOP,
            NOSE_RIDGE_1,
            NOSE_RIDGE_2,
            NOSE_TIP,
            NOSE_RIDGE_TO = NOSE_TIP,
            NOSE_BOTTOM_R,
            NOSE_BOTTOM_FROM = NOSE_BOTTOM_R,
            NOSE_BOTTOM_1,
            NOSE_BOTTOM_2,
            NOSE_BOTTOM_3,
            NOSE_BOTTOM_L,
            NOSE_BOTTOM_TO = NOSE_BOTTOM_L,

            // Eyes [37-42] and [43-48]
            RIGHT_EYE_R,
            RIGHT_EYE_FROM = RIGHT_EYE_R,
            RIGHT_EYE_1,
            RIGHT_EYE_2,
            RIGHT_EYE_L,
            RIGHT_EYE_4,
            RIGHT_EYE_5,
            RIGHT_EYE_TO = RIGHT_EYE_5,
            LEFT_EYE_R,
            LEFT_EYE_FROM = LEFT_EYE_R,
            LEFT_EYE_1,
            LEFT_EYE_2,
            LEFT_EYE_L,
            LEFT_EYE_4,
            LEFT_EYE_5,
            LEFT_EYE_TO = LEFT_EYE_5,

            // Mouth [49-68]
            MOUTH_R,
            MOUTH_OUTER_R = MOUTH_R,
            MOUTH_OUTER_FROM = MOUTH_OUTER_R,
            MOUTH_OUTER_1,
            MOUTH_OUTER_2,
            MOUTH_OUTER_TOP,
            MOUTH_OUTER_4,
            MOUTH_OUTER_5,
            MOUTH_L,
            MOUTH_OUTER_L = MOUTH_L,
            MOUTH_OUTER_7,
            MOUTH_OUTER_8,
            MOUTH_OUTER_BOTTOM,
            MOUTH_OUTER_10,
            MOUTH_OUTER_11,
            MOUTH_OUTER_TO = MOUTH_OUTER_11,
            MOUTH_INNER_R,
            MOUTH_INNER_FROM = MOUTH_INNER_R,
            MOUTH_INNER_1,
            MOUTH_INNER_TOP,
            MOUTH_INNER_3,
            MOUTH_INNER_L,
            MOUTH_INNER_5,
            MOUTH_INNER_BOTTOM,
            MOUTH_INNER_7,
            MOUTH_INNER_TO = MOUTH_INNER_7,

            N_POINTS
        }
        #endregion
        #region Depth Check
        /*
Calculates the average depth for a range of two-dimentional points in face, such that:
   point(n) = face.part(n)
and puts the result in *p_average_depth.
Points for which no depth is available (is 0) are ignored and not factored into the average.
Returns true if an average is available (at least one point has depth); false otherwise.
*/
        private static bool FindDepthFrom(DepthFrame frame, VectorOfPointF landmarks, markup_68 markup_from, markup_68 markup_to, ref float p_average_depth)
        {
            Console.SetCursorPosition(0, 9);
            Console.WriteLine($"depth:{frame.GetDistance(frame.Width / 2, frame.Height / 2)}");
            Console.SetCursorPosition(0, 1);
            float average_depth = 0F;
            uint n_points = 0;
            for (int i = (int)markup_from; i <= (int)markup_to; ++i)
            {
                var pt = landmarks[i];

                if (pt.Y > 0 && pt.Y < frame.Height && pt.X > 0 && pt.X < frame.Width)// если лицо найденные точки лица не все попадают в кадр, тогда отсекаем их
                {
                    var depthInMeters = frame.GetDistance((int)pt.X, (int)pt.Y);

                    if (depthInMeters == 0)
                    {
                        continue;
                    }
                    average_depth += depthInMeters;
                    ++n_points;
                }

            }
            if (n_points == 0)
            {
                return false;
            }
            p_average_depth = average_depth / n_points;
            return true;
        }

        /*
            Returns whether the given 68-point facial landmarks denote the face of a real
            person (and not a picture of one), using the depth data in depth_frame.
            See markup_68 for an explanation of the point topology.
            NOTE: requires the coordinates in face align with those of the depth frame.
        */
        private static bool ValidateFace(DepthFrame frame, VectorOfPointF landmarks)
        {
            // Collect all the depth information for the different facial parts

            // For the ears, only one may be visible -- we take the closer one!
            if (frame == null)
            {
                throw new Exception("Depth map is absent");
                //return false;
            }
            float left_ear_depth = 100F;
            float right_ear_depth = 100F;
            if (!FindDepthFrom(frame, landmarks, markup_68.RIGHT_EAR, markup_68.RIGHT_1, ref right_ear_depth) & !FindDepthFrom(frame, landmarks, markup_68.LEFT_1, markup_68.LEFT_EAR, ref left_ear_depth))
            {
                return false;
            }
            float ear_depth = Math.Min(right_ear_depth, left_ear_depth);

            float chin_depth = 1.0F;
            if (!FindDepthFrom(frame, landmarks, markup_68.CHIN_FROM, markup_68.CHIN_TO, ref chin_depth))
            {
                return false;
            }

            float nose_depth = 1.0F;
            if (!FindDepthFrom(frame, landmarks, markup_68.NOSE_RIDGE_2, markup_68.NOSE_TIP, ref nose_depth))
            {
                return false;
            }

            float right_eye_depth = 1.0F;
            if (!FindDepthFrom(frame, landmarks, markup_68.RIGHT_EYE_FROM, markup_68.RIGHT_EYE_TO, ref right_eye_depth))
            {
                return false;
            }
            float left_eye_depth = 1.0F;
            if (!FindDepthFrom(frame, landmarks, markup_68.LEFT_EYE_FROM, markup_68.LEFT_EYE_TO, ref left_eye_depth))
            {
                return false;
            }
            float eye_depth = Math.Min(left_eye_depth, right_eye_depth);

            float mouth_depth = 1.0F;
            if (!FindDepthFrom(frame, landmarks, markup_68.MOUTH_OUTER_FROM, markup_68.MOUTH_INNER_TO, ref mouth_depth))
            {
                return false;
            }

            // We just use simple heuristics to determine whether the depth information agrees with
            // what's expected: that the nose tip, for example, should be closer to the camera than
            // the eyes.

            // These heuristics are fairly basic but nonetheless serve to illustrate the point that
            // depth data can effectively be used to distinguish between a person and a picture of a
            // person...

            if (nose_depth >= eye_depth)
            {
                return false;
            }
            if (eye_depth - nose_depth > 0.07f)
            {
                return false;
            }
            if (ear_depth <= eye_depth)
            {
                return false;
            }
            if (mouth_depth <= nose_depth)
            {
                return false;
            }
            if (mouth_depth > chin_depth)
            {
                return false;
            }

            // All the distances, collectively, should not span a range that makes no sense. I.e.,
            // if the face accounts for more than 20cm of depth, or less than 2cm, then something's
            // not kosher!
            float[] tempArray = new float[5] { nose_depth, eye_depth, ear_depth, mouth_depth, chin_depth };
            float x = tempArray.Max<float>();
            float n = tempArray.Min<float>();
            if (x - n > 0.20f)
            {
                return false;
            }
            if (x - n < 0.02f)
            {
                return false;
            }

            return true;
        }
        #endregion

    }
}
