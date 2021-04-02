using Emgu.CV;
using Microsoft.Extensions.Configuration;
using System;

namespace MyFaceDNN
{
    class Program
    {
        static void Main(string[] args)
        {
            
            var config = new ConfigurationBuilder()
                .AddJsonFile("config.json", optional: true, reloadOnChange: true)
                .Build();
            //var camIndex = int.Parse(config["CamIndex"]);
            var camIndex = config["CamIndex"];
            var camIndex2 = config["CamIndex2"];
            //var webcam = new WebCam(camIndex);
            var mipi = new MipiCamera(camIndex,camIndex2);
            //var realSense = new RealSenseCam();
            //realSense.GrabFrames();
        }
    }
}
