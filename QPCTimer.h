#pragma once
#include <windows.h>
#include <profileapi.h>


class C_QPCTimer
{
public:
	C_QPCTimer()
	{
		memset( &m_Frequrency, 0, sizeof( LARGE_INTEGER ) );
		memset( &m_LastTime, 0, sizeof( LARGE_INTEGER ) );
	}
	BOOL Initialize()
	{
		return ( !QueryPerformanceFrequency( &m_Frequrency ) ? FALSE : TRUE );
	}
	double GetTime() const
	{
		LARGE_INTEGER now;
		QueryPerformanceCounter( &now );
		return ( (double)now.QuadPart * 1000.0 ) / (double)m_Frequrency.QuadPart;
	}
	double GetElapsedTime()
	{
		LARGE_INTEGER now;
		LONGLONG elapsed;
		QueryPerformanceCounter( &now );
		elapsed = now.QuadPart - m_LastTime.QuadPart;
		m_LastTime = now;
		return ( (double)elapsed * 1000.0 ) / (double)m_Frequrency.QuadPart;
	}

private:
	LARGE_INTEGER m_Frequrency, m_LastTime;
};
