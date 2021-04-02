using Emgu.CV;
using System;
using System.Collections.Generic;
using System.Text;

namespace MyFaceDNN
{
    public class MipiCamera
    {
        static string windowName = "RGB";
        static string windowName2 = "IR";
        public MipiCamera(string camIndex,string camIndex2)
        {

            Console.WriteLine(CvInvoke.BuildInformation);
            CvInvoke.UseOpenCL = true;
            //CvInvoke.UseOptimized = true;
            var faceDetection = new FaceProccesOpenCl();
            VideoCapture videoCapture = new VideoCapture(camIndex, VideoCapture.API.Gstreamer);
            VideoCapture videoCaptureIr = new VideoCapture(camIndex2, VideoCapture.API.Gstreamer);
            //foreach (var i in Enum.GetValues(typeof(VideoCapture.API)))
            //{
            //    videoCapture = new VideoCapture(camIndex, (VideoCapture.API)i);
            //    if (videoCapture.IsOpened)
            //    {
            //        Console.WriteLine($"enum {i}");
            //    }
            //    Console.WriteLine($"enum {i}");
            //}

            CvInvoke.NamedWindow(windowName, Emgu.CV.CvEnum.WindowFlags.AutoSize);
            CvInvoke.NamedWindow(windowName2, Emgu.CV.CvEnum.WindowFlags.AutoSize);

            while (/*CvInvoke.WaitKey(1) != 27 */CvInvoke.GetWindowProperty(windowName, Emgu.CV.CvEnum.WindowPropertyFlags.AutoSize) >= 0)//Esc
            {

                if (!videoCapture.IsOpened)
                {
                    Console.WriteLine("camera could not opened");
                    break;
                }
                try
                {
                    //Console.WriteLine($"camera index:{videoCapture.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.GstreamerQueueLength)}");
                    UMat frame = videoCapture.QueryFrame().GetUMat(Emgu.CV.CvEnum.AccessType.ReadWrite);
                    UMat frameIr = videoCaptureIr.QueryFrame().GetUMat(Emgu.CV.CvEnum.AccessType.ReadWrite);
                    faceDetection.DetectAndRender(frame);
                    CvInvoke.Imshow(windowName, frame);
                    CvInvoke.Imshow(windowName2, frameIr);
                    frame.Dispose();
                    frameIr.Dispose();
                    CvInvoke.WaitKey(1);
                }
                catch (Emgu.CV.Util.CvException cvEx)
                {
                    Console.WriteLine($"cv Exception {cvEx}");
                }
                catch (Exception e)
                {
                    Console.WriteLine(e);
                }
                //faceDetection.DetectAndRender(frame);
                //CvInvoke.Imshow(windowName, frame);
                //frame.Dispose();
                //CvInvoke.WaitKey(5);
            }
        }
    }
}
