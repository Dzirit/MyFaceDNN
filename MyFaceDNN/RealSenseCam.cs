using Emgu.CV;
using Intel.RealSense;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;

namespace MyFaceDNN
{
    public class RealSenseCam
    {

        private static Device dev;
        private static Colorizer colorizer;
        private static Pipeline pipeline;
        //private static RgbPixel badColor = new RgbPixel(255, 0, 0);
        //private static RgbPixel goodColor = new RgbPixel(0, 0, 255);
        private static float depthScale;


        public RealSenseCam()
        {
            using (var ctx = new Context())
            {
                var devices = ctx.QueryDevices();
                Console.WriteLine("There are {0} connected RealSense devices.", devices.Count);
                if (devices.Count == 0) return;
                dev = devices[0];

            }
            Console.SetCursorPosition(0, 3);
            // The colorizer processing block will be used to visualize the depth frames.
            colorizer = new Colorizer();
            // Create and config the pipeline to strem color and depth frames.
            pipeline = new Pipeline();

            var cfg = new Config();
            cfg.EnableStream(Stream.Depth, 640, 480, Format.Z16, 30);

            cfg.EnableStream(Stream.Color, 1280, 720, Format.Bgr8, 30);

            //cfg.EnableStream(Stream.Color, 640, 480, Format.Bgr8, 30);
            depthScale = dev.QuerySensors()[0].DepthScale;//Get depth scale from physical depth sensor
            var profile = pipeline.Start(cfg);
        }
        public void GrabFrames()
        {
            var windowName = "RealSense";
            CvInvoke.NamedWindow(windowName, Emgu.CV.CvEnum.WindowFlags.AutoSize);
            ulong lastFrameNumber = 0;
            var faceProcces = new FaceProccesDNN();
            while (CvInvoke.WaitKey(1) < 0 & CvInvoke.GetWindowProperty(windowName, Emgu.CV.CvEnum.WindowPropertyFlags.AutoSize) >= 0)//Esc
            {
                using (var frames = pipeline.WaitForFrames())
                {


                    Align align = new Align(Stream.Color).DisposeWith(frames);
                    Frame aligned = align.Process(frames).DisposeWith(frames);
                    FrameSet alignedframeset = aligned.As<FrameSet>().DisposeWith(frames);
                    var colorFrame = alignedframeset.ColorFrame.DisposeWith(frames);
                    var depthFrame = alignedframeset.DepthFrame.DisposeWith(frames);
                    if (colorFrame.Number == lastFrameNumber)
                    {
                        continue;
                    }
                    lastFrameNumber = colorFrame.Number;
                    //Console.WriteLine($"depth:{depthFrame.GetDistance(depthFrame.Width / 2, depthFrame.Height / 2)}");
                    //var image = FrameToMat(colorFrame);
                    var image = new Mat(new Size(colorFrame.Width, colorFrame.Height), Emgu.CV.CvEnum.DepthType.Cv8U, 3, colorFrame.Data, 0);
                    faceProcces.DetectAndRender(image, depthFrame);
                    CvInvoke.Imshow(windowName, image);
                }
            }
        }
        private Mat FrameToMat(Frame f)
        {
            var vf = f.As<VideoFrame>();
            if (f.Profile.Format == Format.Bgr8)
            {
                return new Mat(new Size(vf.Width, vf.Height), Emgu.CV.CvEnum.DepthType.Cv8U, 3, f.Data, 0);
            }
            else if (f.Profile.Format == Format.Rgb8)
            {
                var rgb = new Mat(new Size(vf.Width, vf.Height), Emgu.CV.CvEnum.DepthType.Cv8U, 3, f.Data, 0);
                var bgr = new Mat();
                CvInvoke.CvtColor(rgb, bgr, Emgu.CV.CvEnum.ColorConversion.Rgb2Bgr);
                return bgr;
            }
            else if (f.Profile.Format == Format.Z16)
            {
                return new Mat(new Size(vf.Width, vf.Height), Emgu.CV.CvEnum.DepthType.Cv16U, 3, f.Data, 0);
            }
            else if (f.Profile.Format == Format.Y8)
            {
                return new Mat(new Size(vf.Width, vf.Height), Emgu.CV.CvEnum.DepthType.Cv8U, 3, f.Data, 0);
            }
            else if (f.Profile.Format == Format.Disparity32)
            {
                return new Mat(new Size(vf.Width, vf.Height), Emgu.CV.CvEnum.DepthType.Cv32F, 3, f.Data, 0);
            }
            else
            {
                throw new Exception("Не поддерживаемый формат");
            }
        }
    }
}
