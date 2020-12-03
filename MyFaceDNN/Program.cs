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
            var camIndex = int.Parse(config["CamIndex"]);
            //var webcam = new WebCam(camIndex);
            var realSense = new RealSenseCam();
            realSense.GrabFrames();
        }
    }
}
